import h5py
with h5py.File('skin_disease_model.h5', 'r') as f:
    if 'model_config' in f.attrs:
        print('model_config exists')
    else:
        print('no model_config in file attrs')
    # print top keys
    print('keys:', list(f.keys()))
    # if keras metadata, show
    if 'model_config' in f.attrs:
        import json
        raw = f.attrs['model_config']
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8')
        config = json.loads(raw)
        print('model config type:', config.get('class_name'))
        if 'config' in config and 'layers' in config['config']:
            for i,l in enumerate(config['config']['layers']):
                if l.get('class_name','').startswith('Conv'):
                    print('conv layer', i, l['config'].get('name'), 'input_shape', l['config'].get('batch_input_shape'), 'filters', l['config'].get('filters'))
        # print first layer config
        print('first layer config class', config['config']['layers'][0]['class_name'])
    else:
        print('not sequential?')
