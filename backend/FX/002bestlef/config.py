{
    "name": "002bestlef",
    "description": "Pixel sorting effect applied within segmentation mask, with options for thresholds, sorting direction, and visual adjustments.",
    "defaultParams": {
        "threshold_min": {
            "type": "int",
            "value": 0,
            "min": 0,
            "max": 255,
            "step": 1
        },
        "threshold_max": {
            "type": "int",
            "value": 255,
            "min": 0,
            "max": 255,
            "step": 1
        },
        "angle": {
            "type": "float",
            "value": 0,
            "min": 0,
            "max": 360,
            "step": 1
        },
        "contrast": {
            "type": "float",
            "value": 1.0,
            "min": 0.5,
            "max": 2.0,
            "step": 0.1
        },
        "saturation": {
            "type": "float",
            "value": 1.0,
            "min": 0.5,
            "max": 2.0,
            "step": 0.1
        },
        "resolution": {
            "type": "int",
            "value": 1,
            "min": 1,
            "max": 10,
            "step": 1
        },
        "noise_amount": {
            "type": "int",
            "value": 0,
            "min": 0,
            "max": 100,
            "step": 1
        },
        "noise_scale": {
            "type": "float",
            "value": 1.0,
            "min": 0.1,
            "max": 10.0,
            "step": 0.1
        },
        "sorting_direction": {
            "type": "int",
            "value": 0,
            "min": 0,
            "max": 1,
            "step": 1
        },
        "sorting_style": {
            "type": "int",
            "value": 0,
            "min": 0,
            "max": 2,
            "step": 1
        }
    }
}