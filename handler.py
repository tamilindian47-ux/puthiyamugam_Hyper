"""Runpod entry point: native multi-identity jobs and persistent chunk delivery."""
import base64
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time
import traceback
import uuid
from PIL import Image, ImageOps
import runpod
from scanner import download

BUILD_ID='pm-multi-native-v1'
FF_DIR='/opt/facefusion'
FF_PYTHON='/opt/ffenv/bin/python'
PIPELINES={
    'hs_a':('hyperswap_1a_256',False),'hs_a_expr':('hyperswap_1a_256',True),
    'hs_b':('hyperswap_1b_256',False),'hs_b_expr':('hyperswap_1b_256',True),
    'hs_c':('hyperswap_1c_256',False),'hs_c_expr':('hyperswap_1c_256',True),
}
CHUNK_BYTES=1024*1024


def sha256(path):
    with open(path,'rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hash_stream(handle)


def hash_stream(handle):
    value=hashlib.sha256()
    for chunk in iter(lambda:handle.read(1024*1024),b''):
        value.update(chunk)
    return value.hexdigest()


def results_root():
    volume=Path('/runpod-volume')
    if not volume.is_dir():
        raise RuntimeError('Attach a shared Runpod network volume mounted at /runpod-volume. It is required for reliable output downloads across workers; no third-party hosting is used.')
    root=Path(os.getenv('RESULTS_DIR','/runpod-volume/pm-results'))
    if volume.resolve() not in root.resolve().parents:
        raise RuntimeError('RESULTS_DIR must be inside /runpod-volume.')
    root.mkdir(parents=True,exist_ok=True)
    return root


def store_result(path,media_type,mime,metrics):
    root=results_root()
    artifact=uuid.uuid4().hex
    folder=root/artifact
    folder.mkdir()
    destination=folder/('result'+Path(path).suffix)
    shutil.copyfile(path,destination)
    size=destination.stat().st_size
    manifest={'status':'success','delivery':'chunks','artifact_id':artifact,
              'filename':destination.name,'mime_type':mime,'media_type':media_type,
              'size_bytes':size,'chunk_bytes':CHUNK_BYTES,
              'chunk_count':math.ceil(size/CHUNK_BYTES),'sha256':sha256(destination),
              'expires_at':int(time.time())+int(os.getenv('RESULT_TTL_SECONDS','86400')),
              'metrics':metrics,'build_id':BUILD_ID}
    temporary=folder/'manifest.tmp'
    temporary.write_text(json.dumps(manifest))
    temporary.replace(folder/'manifest.json')
    return manifest


def read_manifest(artifact):
    if not isinstance(artifact,str) or not re.fullmatch('[a-f0-9]{32}',artifact):
        raise ValueError('Invalid artifact ID.')
    folder=results_root()/artifact
    file=folder/'manifest.json'
    if not file.is_file():
        raise ValueError('Result not found on the shared volume. Check that every worker mounts the SAME network volume.')
    manifest=json.loads(file.read_text())
    if time.time()>manifest['expires_at']:
        raise ValueError('Stored result has expired.')
    return folder,manifest


def cleanup_expired():
    root=results_root()
    for file in list(root.glob('*/manifest.json'))[:1000]:
        try:
            data=json.loads(file.read_text())
            if time.time()>data['expires_at']+3600:
                shutil.rmtree(file.parent)
        except (OSError,ValueError,KeyError):
            continue


def fetch_chunk(job):
    folder,manifest=read_manifest(job.get('artifact_id'))
    index=job.get('index')
    if type(index) is not int or not 0<=index<manifest['chunk_count']:
        raise ValueError('Invalid chunk index.')
    with open(folder/manifest['filename'],'rb') as handle:
        handle.seek(index*manifest['chunk_bytes'])
        data=handle.read(manifest['chunk_bytes'])
    expected=min(manifest['chunk_bytes'],manifest['size_bytes']-index*manifest['chunk_bytes'])
    if len(data)!=expected:
        raise RuntimeError('Stored output is incomplete.')
    return {'status':'success','index':index,'size_bytes':len(data),
            'sha256':hashlib.sha256(data).hexdigest(),
            'base64':base64.b64encode(data).decode('ascii')}


def progress(event,**data):
    print('PROGRESS '+json.dumps(data),flush=True)
    try:
        runpod.serverless.progress_update(event,data)
    except Exception as error:
        print('Progress update unavailable: '+str(error)[:200],flush=True)


def image_to_png(source,destination):
    with Image.open(source) as image:
        image=ImageOps.exif_transpose(image)
        if image.width*image.height>50_000_000:
            raise ValueError('Image exceeds 50 megapixels.')
        image.convert('RGB').save(destination,'PNG')


def probe_media(path,directory):
    try:
        with Image.open(path) as image:
            animated=bool(getattr(image,'is_animated',False))
            if not animated:
                destination=str(Path(directory)/'target.png')
                image_to_png(path,destination)
                return destination,'image',{'width':image.width,'height':image.height}
            is_gif=image.format=='GIF'
    except (OSError,ValueError):
        is_gif=False
    completed=subprocess.run(['ffprobe','-v','error','-select_streams','v:0',
                              '-show_entries','stream=width,height,avg_frame_rate,r_frame_rate:stream_side_data=rotation:format=duration',
                              '-of','json',str(path)],capture_output=True,text=True,timeout=60)
    if completed.returncode:
        raise ValueError('Cannot decode media. Use a direct media file URL, not a webpage.')
    info=json.loads(completed.stdout)
    streams=info.get('streams',[])
    if not streams:
        raise ValueError('No image/video stream found.')
    stream=streams[0]
    width,height=int(stream['width']),int(stream['height'])
    rotation=next((float(item.get('rotation',0)) for item in stream.get('side_data_list',[]) if 'rotation' in item),0)
    if abs(round(rotation))%180==90:
        width,height=height,width
    if width*height>int(os.getenv('MAX_VIDEO_PIXELS',str(3840*2160))):
        raise ValueError('Video resolution exceeds MAX_VIDEO_PIXELS.')
    rate=None
    for name in ('avg_frame_rate','r_frame_rate'):
        try:
            candidate=Fraction(stream.get(name,'0/1'))
            if 0<float(candidate)<=240:
                rate=str(candidate);break
        except (ValueError,ZeroDivisionError):
            pass
    if is_gif:
        rate=os.getenv('GIF_FPS','25')
    if not rate:
        raise ValueError('Cannot determine frame rate.')
    duration=float(info.get('format',{}).get('duration',0) or 0)
    if duration>float(os.getenv('MAX_DURATION_SECONDS','600')):
        raise ValueError('Media exceeds MAX_DURATION_SECONDS.')
    return str(path),'gif' if is_gif else 'video',{'width':width,'height':height,'fps':float(Fraction(rate)),'duration':duration}


def run_bridge(config,directory,event):
    path=Path(directory)/'config.json'
    config['result_json']=str(Path(directory)/'bridge-result.json')
    path.write_text(json.dumps(config))
    log_path=Path(directory)/'processing.log'
    limit=int(os.getenv('JOB_TIMEOUT_SECONDS','3600'))
    started=time.monotonic()
    with open(log_path,'wb') as log:
        child=subprocess.Popen([FF_PYTHON,'-u','/app/ff_bridge.py',str(path)],
                               cwd=FF_DIR,stdout=log,stderr=subprocess.STDOUT,
                               start_new_session=True,env={**os.environ,'PYTHONUNBUFFERED':'1'})
        previous=''
        try:
            while child.poll() is None:
                time.sleep(3)
                if time.monotonic()-started>limit:
                    raise TimeoutError('Processing exceeded JOB_TIMEOUT_SECONDS.')
                text=log_path.read_text(errors='replace')[-12000:]
                lines=[line[len('PM_PROGRESS:'):] for line in text.splitlines() if line.startswith('PM_PROGRESS:')]
                if lines and lines[-1]!=previous:
                    previous=lines[-1]
                    progress(event,**json.loads(previous))
            details=log_path.read_text(errors='replace')[-12000:]
            print(details,flush=True)
            if child.returncode:
                raise RuntimeError('FaceFusion processing failed:\n'+details[-5000:])
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
    if not Path(config['result_json']).is_file():
        raise RuntimeError('Processor exited without a result manifest.')
    return json.loads(Path(config['result_json']).read_text())


def handler(event):
    identifier=event.get('id','unknown')
    print(f'JOB START {identifier} BUILD={BUILD_ID}',flush=True)
    try:
        job=event.get('input',{})
        if not isinstance(job,dict):
            raise ValueError('Input must be a JSON object.')
        action=job.get('action','swap')
        if action=='fetch_chunk':
            return fetch_chunk(job)
        if action=='result_manifest':
            return read_manifest(job.get('artifact_id'))[1]
        if action not in ('detect','swap'):
            raise ValueError('Unknown action.')
        if action=='swap':
            results_root()
            cleanup_expired()
        pipeline=job.get('pipeline','hs_a')
        if pipeline not in PIPELINES:
            raise ValueError('Unknown pipeline. Use the new HTML.')
        model,restore=PIPELINES[pipeline]
        factor=int(job.get('expression_factor',60))
        if not 0<=factor<=100:
            raise ValueError('Expression factor must be 0–100.')
        with tempfile.TemporaryDirectory(prefix='pm_multi_') as directory:
            incoming=Path(directory)/'input.bin'
            progress(event,stage='downloading')
            original_bytes=download(job.get('media'),incoming)
            media_hash=sha256(incoming)
            if action=='swap' and job.get('media_sha256')!=media_hash:
                raise ValueError('Target media changed since scanning. Scan this exact file/link again.')
            target,kind,probe=probe_media(incoming,directory)
            config={'action':action,'target':target,'media_type':kind,'probe':probe,
                    'model':model,'restore':restore if action=='swap' else False,
                    'expression_factor':factor}
            if action=='detect':
                result=run_bridge(config,directory,event)
                result.update(media_sha256=media_hash,input_bytes=original_bytes,media_type=kind,probe=probe,build_id=BUILD_ID)
                return result
            identities=job.get('identities')
            mappings=job.get('mappings')
            source_inputs=job.get('sources')
            if not isinstance(identities,list) or not 1<=len(identities)<=40:
                raise ValueError('Scan faces first.')
            if not isinstance(mappings,dict) or not mappings or not isinstance(source_inputs,dict):
                raise ValueError('Assign at least one replacement photo.')
            ids=[str(item['character_id']) for item in identities]
            if len(set(ids))!=len(ids) or any(str(key) not in ids for key in mappings):
                raise ValueError('Invalid/duplicate identity mapping.')
            if any(not isinstance(key,str) or key not in source_inputs for key in mappings.values()):
                raise ValueError('A mapped replacement photo is missing.')
            sources={}
            for index,key in enumerate(dict.fromkeys(mappings.values())):
                raw=Path(directory)/f'source-{index}.bin'
                download(source_inputs[key],raw)
                destination=Path(directory)/f'source-{index}.png'
                image_to_png(raw,destination)
                sources[key]=str(destination)
            output=Path(directory)/('result.png' if kind=='image' else 'result.mp4')
            config.update(sources=sources,identities=identities,mappings=mappings,
                          identity_space=job.get('identity_space'),output=str(output))
            result=run_bridge(config,directory,event)
            output_format=job.get('output_format','mp4')
            if output_format not in ('mp4','gif'):
                raise ValueError('Output format must be mp4 or gif.')
            mime='image/png' if kind=='image' else 'video/mp4'
            output_kind='image' if kind=='image' else 'video'
            if kind!='image' and output_format=='gif':
                progress(event,stage='encoding_gif')
                gif=Path(directory)/'result.gif'
                command=['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(output),
                         '-filter_complex','split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
                         '-loop','0',str(gif)]
                completed=subprocess.run(command,capture_output=True,timeout=int(os.getenv('JOB_TIMEOUT_SECONDS','3600')))
                if completed.returncode:
                    raise RuntimeError('GIF encoding failed: '+completed.stderr.decode(errors='replace')[-2000:])
                output=gif;mime='image/gif';output_kind='gif'
            if not output.is_file() or output.stat().st_size==0:
                raise RuntimeError('No output file was produced.')
            progress(event,stage='saving_result',output_bytes=output.stat().st_size)
            result.update(pipeline=pipeline,model=model,input_bytes=original_bytes,
                          expression_factor=factor if restore else 0)
            return store_result(output,output_kind,mime,result)
    except Exception:
        traceback.print_exc()
        # Raise instead of disguising application failures as completed successes.
        raise
    finally:
        print('JOB END '+identifier,flush=True)


def configure_assets():
    destination=os.getenv('FF_ASSETS_DIR','').strip()
    if not destination:
        return
    path=Path(destination)
    if not path.parent.is_dir():
        raise RuntimeError('FF_ASSETS_DIR parent volume is not mounted.')
    path.mkdir(parents=True,exist_ok=True)
    link=Path(FF_DIR)/'.assets'
    if link.is_symlink() and link.resolve()==path.resolve():
        return
    if link.exists() or link.is_symlink():
        raise RuntimeError('FaceFusion .assets already exists; leave FF_ASSETS_DIR empty or migrate it first.')
    link.symlink_to(path,target_is_directory=True)

if __name__=='__main__':
    configure_assets()
    print('READY '+BUILD_ID+'; native multi-person identity matching; no runtime source rewriting.',flush=True)
    runpod.serverless.start({'handler':handler})
