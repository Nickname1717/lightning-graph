import src.data.components

def custom_preprocess(data, preprocess_type):
    preprocess_func_name = f'proprecess_{preprocess_type}'
    if hasattr(src.data.components, preprocess_func_name):
        preprocess_func = getattr(src.data.components, preprocess_func_name)
        if callable(preprocess_func):
            return preprocess_func(data)

    raise ValueError(f"Unsupported preprocess_type: {preprocess_type}")

def proprecess_addone(data):
    return data + 1

def proprecess_addtwo(data):
    return data + 2
