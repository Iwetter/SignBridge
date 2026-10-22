"""DirectShow device identities, with indices resolved only at open time."""
import re


def device_kind(name, path):
    text = (name + ' ' + path).casefold()
    if re.search(r'virtual|synthetic|\bobs\b|gopro webcam|manycam|droidcam|iriun|snap camera|xsplit|camo camera|ndi video', text):
        return 'virtual'
    if any(token in text for token in ('usb#', 'pci#', 'acpi#', 'vid_', 'usb\\', 'pci\\')):
        return 'physical'
    return 'unknown'


def device_record(index, name, path=''):
    name, path = str(name).strip(), str(path or '').strip()
    if not name:
        raise ValueError('DirectShow returned a camera without a device name.')
    return dict(index=int(index), name=name, path=path, backend='DSHOW',
                device_id=('path:' + path.casefold()) if path else ('name:' + name.casefold()),
                kind=device_kind(name,path))


def enumerate_dshow_devices():
    # Explicit backend yields indices in DirectShow order, not MSMF or a sorted PnP list.
    import cv2
    from cv2_enumerate_cameras import enumerate_cameras
    return [device_record(item.index,item.name,item.path) for item in enumerate_cameras(cv2.CAP_DSHOW)]


def identity(device):
    return {key:device[key] for key in ('device_id','name','path','backend','kind')}


def resolve_device(devices, preferred=None, *, explicit=False):
    preferred = preferred or {}
    requested = preferred.get('device_id')
    if not explicit and preferred.get('kind') in ('virtual','unknown'):
        requested = None
    if requested:
        matches = [d for d in devices if d['device_id']==requested]
        if len(matches)>1:
            raise ValueError('Multiple cameras have the same name and no unique device path. Cannot select one safely.')
        if not matches:
            raise ValueError(f'Selected camera {preferred.get("name") or requested} is disconnected. Refresh cameras and select a device by name.')
        device = matches[0]
        if explicit or device['kind']=='physical':
            return dict(device)
        # A persisted virtual/unknown device is not an automatic physical-camera default.
    physical = [d for d in devices if d['kind']=='physical']
    if physical:
        return dict(sorted(physical,key=lambda d:(d['name'].casefold(),d['device_id']))[0])
    raise ValueError('No physical webcam identified. Refresh cameras and explicitly choose a named device. Virtual cameras are never selected automatically.')
