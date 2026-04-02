"""
@Author: teddy1565
@Description: MFA
@Title: MFA
@Nickname: MFA
"""

from . import node

NODE_CLASS_MAPPINGS = {
    "MFA_AudioToText": node.MFA_AudioToText
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MFA_AudioToText": "MFA Audio To Segments"
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS"
]