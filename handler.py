"""No-volume multi-person handler, with version-compatible FFprobe metadata reads."""
import base64
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import traceback
from PIL import Image, ImageOps
import runpod
from scanner import download

BUILD_ID='pm-multi-inline-v3'
PATCH_ID='probe-compat-1'
FF_DIR='/opt/facefusion'
FF_PYTHON='/opt/ffenv/bin/python'
MAX_JSON_BYTES=min(9_000_000,int(os.getenv('MAX_INLINE_JSON_BYTES','9000000')))
PIPELINES={
    'hs_a':('hyperswap_1a_256',False),'hs_a_expr':('hyperswap_1a_256',True),
    'hs_b':('hyperswap_1b_256',False),'hs_b_expr':('hyperswap_1b_256',True),
    'hs_c':('hyperswap_1c_256',False),'hs_c_expr':('hyperswap_1c_256',True),
}
ACTIONS=('capabilities','detect','swap')


def sha256(path):
    value=hashlib.sha256()
    with open(path,'rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def bounded_result(result):
    size=len(json.dumps(result,ensure_ascii=True).encode('utf-8'))
    if size>MAX_JSON_BYTES:
        raise ValueError(
            f'Result JSON is {size/1048576:.2f} MiB; direct-delivery budget is '
            f'{MAX_JSON_BYTES/1048576:.2f} MiB. No result was stored. '
            'Use shorter/lower-resolution media or explicitly choose a smaller MP4 output setting.'
        )
    return result


def progress(event,**data):
    print('PROGRESS '+json.dumps(data),flush=True)
    try:
        runpod.serverless.progress_update(event,data)
    except Exception as error:
        print('Progress update unavailable: '+str(error)[:200],flush=True)


def image_to_png(source,destination):
    with Image.open(source) as image:
        if image.width*image.height>50_000_000:
            raise ValueError('Image exceeds 50 megapixels.')
        if getattr(image,'is_animated',False):
            raise ValueError('Replacement photos must be still images.')
        image=ImageOps.exif_transpose(image)
        image.convert('RGB').save(destination,'PNG')


def finite_number(value,default=0.0):
    try:
        value=float(value)
        return value if math.isfinite(value) else default
    except (TypeError,ValueError):
        return default


def probe_media(path,directory):
    path=Path(path)
    if not path.is_file() or path.stat().st_size==0:
        raise ValueError('Received an empty media file.')
    with open(path,'rb') as handle:
        prefix=handle.read(64)
    beginning=prefix.lstrip().lower()
    if beginning.startswith((b'<!doctype html',b'<html',b'<?xml')):
        raise ValueError('Received a webpage/XML response instead of media. Use a direct media-file URL.')
    still,is_gif=False,False
    try:
        with Image.open(path) as image:
            animated=bool(getattr(image,'is_animated',False))
            still=not animated
            is_gif=animated and image.format=='GIF'
    except OSError:
        pass
    if still:
        target=str(Path(directory)/'target.png')
        image_to_png(path,target)
        with Image.open(target) as image:
            return target,'image',{'width':image.width,'height':image.height}

    # No version-specific -show_entries section aliases.
    command=[
        'ffprobe','-v','error','-select_streams','v:0',
        '-show_streams','-show_format','-of','json',str(path),
    ]
    completed=subprocess.run(command,capture_output=True,text=True,timeout=60)
    if completed.returncode:
        details=(completed.stderr or completed.stdout or 'No FFprobe diagnostic returned.').strip()[-2400:]
        raise ValueError(
            f'Media probe failed (FFprobe exit {completed.returncode}; '
            f'{path.stat().st_size} input bytes; header {prefix[:16].hex()}).\n'
            'FFprobe diagnostic:\n'+details
        )
    try:
        info=json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError('FFprobe returned invalid metadata JSON: '+completed.stdout[:500]) from error
    streams=[item for item in info.get('streams',[]) if item.get('codec_type')=='video']
    if not streams:
        raise ValueError('No video/image stream found. The file may contain only audio.')
    stream=streams[0]
    width,height=int(stream.get('width',0)),int(stream.get('height',0))
    if min(width,height)<=0:
        raise ValueError('FFprobe could not determine media dimensions.')
    side=stream.get('side_data_list',[])
    rotation=next((finite_number(item['rotation']) for item in side if 'rotation' in item),None)
    if rotation is None:
        rotation=finite_number(stream.get('tags',{}).get('rotate',0))
    if abs(round(rotation))%180==90:
        width,height=height,width
    if width*height>int(os.getenv('MAX_VIDEO_PIXELS',str(3840*2160))):
        raise ValueError('Video exceeds MAX_VIDEO_PIXELS.')
    rate=None
    for key in ('avg_frame_rate','r_frame_rate'):
        try:
            candidate=Fraction(stream.get(key,'0/1'))
            if 0<float(candidate)<=240:
                rate=candidate
                break
        except (TypeError,ValueError,ZeroDivisionError):
            pass
    if is_gif:
        rate=Fraction(os.getenv('GIF_FPS','25'))
    if rate is None or not 0<float(rate)<=240:
        raise ValueError('Cannot determine a valid frame rate.')
    duration=finite_number(info.get('format',{}).get('duration'),finite_number(stream.get('duration')))
    if duration>float(os.getenv('MAX_DURATION_SECONDS','600')):
        raise ValueError('Media exceeds MAX_DURATION_SECONDS.')
    print('MEDIA_PROBE '+json.dumps({'bytes':path.stat().st_size,'width':width,'height':height,'fps':float(rate),'format':info.get('format',{}).get('format_name'),'patch':PATCH_ID}),flush=True)
    return str(path),'gif' if is_gif else 'video',{'width':width,'height':height,'fps':float(rate),'duration':duration}


def tail(path,size=12000):
    with open(path,'rb') as handle:
        handle.seek(0,2)
        handle.seek(max(0,handle.tell()-size))
        return handle.read().decode('utf-8',errors='replace')


def run_bridge(config,directory,event):
    path=Path(directory)/'config.json'
    config['result_json']=str(Path(directory)/'bridge-result.json')
    path.write_text(json.dumps(config),encoding='utf-8')
    log_path=Path(directory)/'processing.log'
    started=time.monotonic()
    timeout=int(os.getenv('JOB_TIMEOUT_SECONDS','3600'))
    environment={**os.environ,'PYTHONUNBUFFERED':'1','OUTPUT_CRF':str(config.get('output_crf',18))}
    with open(log_path,'wb') as log:
        child=subprocess.Popen(
            [FF_PYTHON,'-u','/app/ff_bridge.py',str(path)],cwd=FF_DIR,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=environment,
        )
        previous=''
        try:
            while child.poll() is None:
                time.sleep(3)
                if time.monotonic()-started>timeout:
                    raise TimeoutError('Processing exceeded JOB_TIMEOUT_SECONDS.')
                updates=[line[len('PM_PROGRESS:'):] for line in tail(log_path).splitlines() if line.startswith('PM_PROGRESS:')]
                if updates and updates[-1]!=previous:
                    try:
                        update=json.loads(updates[-1])
                    except json.JSONDecodeError:
                        continue
                    previous=updates[-1]
                    progress(event,**update)
            details=tail(log_path)
            print(details,flush=True)
            if child.returncode:
                raise RuntimeError('FaceFusion processing failed:\n'+details[-5000:])
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait()
    result_file=Path(config['result_json'])
    if not result_file.is_file():
        raise RuntimeError('Processor exited without a result manifest.')
    return json.loads(result_file.read_text())


def handler(event):
    job_id=str(event.get('id','unknown'))
    print(f'JOB START {job_id} BUILD={BUILD_ID} PATCH={PATCH_ID}',flush=True)
    try:
        job=event.get('input',{})
        if not isinstance(job,dict):
            raise ValueError('Input must be a JSON object.')
        action=job.get('action','swap')
        print('ACTION='+repr(action),flush=True)
        if action not in ACTIONS:
            raise ValueError(f'Unsupported action {action!r}. Use the no-volume pm-multi-inline-v3 HTML. Supported: '+', '.join(ACTIONS))
        if action=='capabilities':
            return {'status':'success','build_id':BUILD_ID,'patch_id':PATCH_ID,
                    'actions':list(ACTIONS),'network_volume_required':False,
                    'delivery':'inline','max_response_json_bytes':MAX_JSON_BYTES}
        pipeline=job.get('pipeline','hs_a')
        if pipeline not in PIPELINES:
            raise ValueError('Unknown pipeline. Use the matching HTML.')
        model,restore=PIPELINES[pipeline]
        factor=int(job.get('expression_factor',60))
        if not 0<=factor<=100:
            raise ValueError('Expression factor must be 0–100.')
        crf=int(job.get('output_crf',18))
        if crf not in (18,23,28):
            raise ValueError('Output CRF must be 18, 23, or 28.')
        output_format=job.get('output_format','mp4')
        if output_format not in ('mp4','gif'):
            raise ValueError('Animated output must be MP4 or GIF.')
        if not isinstance(job.get('media'),str):
            raise ValueError('Media must be a direct URL or base64 string; upload references are not supported.')
        with tempfile.TemporaryDirectory(prefix='pm_inline_') as directory:
            incoming=Path(directory)/'input.bin'
            progress(event,stage='downloading')
            download(job['media'],incoming)
            if not incoming.is_file():
                raise ValueError('Downloader did not create the media file.')
            input_bytes=incoming.stat().st_size
            media_hash=sha256(incoming)
            if action=='swap' and job.get('media_sha256')!=media_hash:
                raise ValueError('Target changed since scanning. Scan the exact file/URL again.')
            target,kind,probe=probe_media(incoming,directory)
            config={'action':action,'target':target,'media_type':kind,'probe':probe,
                    'model':model,'restore':restore if action=='swap' else False,
                    'expression_factor':factor,'output_crf':crf}
            if action=='detect':
                result=run_bridge(config,directory,event)
                result.update(media_sha256=media_hash,input_bytes=input_bytes,media_type=kind,
                              probe=probe,build_id=BUILD_ID,patch_id=PATCH_ID)
                return bounded_result(result)
            identities=job.get('identities')
            mappings=job.get('mappings')
            source_inputs=job.get('sources')
            if not isinstance(identities,list) or not 1<=len(identities)<=40:
                raise ValueError('Scan faces first.')
            if not all(isinstance(item,dict) and 'character_id' in item for item in identities):
                raise ValueError('Invalid identity records.')
            if not isinstance(mappings,dict) or not mappings or not isinstance(source_inputs,dict):
                raise ValueError('Assign at least one replacement photo.')
            ids=[str(item['character_id']) for item in identities]
            if len(set(ids))!=len(ids) or any(str(key) not in ids for key in mappings):
                raise ValueError('Invalid/duplicate identity mapping.')
            if any(not isinstance(key,str) or key not in source_inputs for key in mappings.values()):
                raise ValueError('A replacement photo is missing.')
            sources={}
            for index,key in enumerate(dict.fromkeys(mappings.values())):
                if not isinstance(source_inputs[key],str):
                    raise ValueError('Replacement photos must use URLs/base64, not upload references.')
                raw=Path(directory)/f'source-{index}.bin'
                download(source_inputs[key],raw)
                png=Path(directory)/f'source-{index}.png'
                image_to_png(raw,png)
                sources[key]=str(png)
            output=Path(directory)/('result.png' if kind=='image' else 'result.mp4')
            config.update(sources=sources,identities=identities,mappings=mappings,
                          identity_space=job.get('identity_space'),output=str(output))
            metrics=run_bridge(config,directory,event)
            if metrics.get('status')=='failed' or metrics.get('error'):
                raise RuntimeError(metrics.get('error','Processor failed.'))
            mime,output_kind=('image/png','image') if kind=='image' else ('video/mp4','video')
            if kind!='image' and output_format=='gif':
                progress(event,stage='encoding_gif')
                gif=Path(directory)/'result.gif'
                completed=subprocess.run([
                    'ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(output),
                    '-filter_complex','split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse',
                    '-loop','0',str(gif),
                ],capture_output=True,timeout=int(os.getenv('JOB_TIMEOUT_SECONDS','3600')))
                if completed.returncode:
                    raise RuntimeError('GIF encoding failed: '+completed.stderr.decode(errors='replace')[-2000:])
                output,mime,output_kind=gif,'image/gif','gif'
            if not output.is_file() or output.stat().st_size==0:
                raise RuntimeError('No output file produced.')
            size=output.stat().st_size
            metrics.update(pipeline=pipeline,model=model,input_bytes=input_bytes,
                           expression_factor=factor if restore else 0,output_crf=crf)
            result={'status':'success','delivery':'inline','build_id':BUILD_ID,'patch_id':PATCH_ID,
                    'job_id':job_id,'filename':output.name,'mime_type':mime,'media_type':output_kind,
                    'size_bytes':size,'sha256':sha256(output),'metrics':metrics,'base64':''}
            estimated=len(json.dumps(result).encode('utf-8'))+4*((size+2)//3)
            if estimated>MAX_JSON_BYTES:
                raise ValueError(
                    f'Output is {size/1048576:.2f} MiB; its base64 JSON exceeds the '
                    f'{MAX_JSON_BYTES/1048576:.2f} MiB direct-delivery budget. No video was stored. '
                    'Use shorter/lower-resolution media, MP4 instead of GIF, or explicitly select CRF 23/28. '
                    'No automatic quality reduction was applied.'
                )
            progress(event,stage='returning_output',output_bytes=size)
            result['base64']=base64.b64encode(output.read_bytes()).decode('ascii')
            return bounded_result(result)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        print('JOB END '+job_id,flush=True)


if __name__=='__main__':
    print('READY '+BUILD_ID+' PATCH='+PATCH_ID+'; no network volume; no runtime source rewriting.',flush=True)
    runpod.serverless.start({'handler':handler})
