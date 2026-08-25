import json

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles common non-serializable types."""
    
    def default(self, obj):
        if hasattr(obj, '__class__'):
            class_name = obj.__class__.__name__
            if 'Role' in class_name:
                return f"{class_name}({getattr(obj, 'data_type', None)})"
            if hasattr(obj, 'item'):
                return obj.item()
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
        return super().default(obj)