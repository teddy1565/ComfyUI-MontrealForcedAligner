import os
import time
import json
import math
import random
import numpy as np
import uuid
import torchaudio
import torch
import pathlib


import nodes
import folder_paths
import comfy.utils
import comfy_execution
from server import PromptServer
import tgt
from montreal_forced_aligner.alignment.pretrained import PretrainedAligner
import comfy.model_management as mm



MAIN_CATEGORY = "MontrealForcedAligner/Main"

class MFA_AudioToText:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "forceInput": True,
                }),
                "dubbing_draft": ("STRING", {
                    "forceInput": True
                }),
                "ACOUSTIC_MODEL_PATH": ("STRING", {
                    "tooltip": "MFA ACOUSTIC_MODEL"
                }),
                "DICTIONARY_PATH": ("STRING", {
                    "tooltip": "MFA DICTIONARY"
                }),
                "segments_size": ("INT", {
                    "min": 1,
                    "default": 1,
                    "tooltip": "how many char be a segments"
                }),
            },
            "optional": {
                "show_verbose": ("BOOLEAN", {
                    "default": False
                })
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            }
        }
    
    RETURN_TYPES = ("whisper_alignment",)
    RETURN_NAMES = ("segments_alignment",)
    CATEGORY = f'{MAIN_CATEGORY}'
    FUNCTION = "audioToString"
    DESCRIPTION = \
    """
    Montreal Forced Aligner Model, In windows, model path usually is: C:/user/<user_name>/Documents/MFA
    """
    
    def audioToString(self, audio, dubbing_draft, ACOUSTIC_MODEL_PATH, DICTIONARY_PATH, segments_size=1, show_verbose=False, unique_id=0):
        
        ACOUSTIC_MODEL_PATH = str(pathlib.Path(ACOUSTIC_MODEL_PATH).resolve())
        DICTIONARY_PATH = str(pathlib.Path(DICTIONARY_PATH).resolve())
        temp_dir = folder_paths.get_temp_directory()
        out_temp_dir = os.path.join(temp_dir, f"mfa-output-data")
        temp_dir = os.path.join(temp_dir, f"mfa-corpus-data")
        
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(out_temp_dir, exist_ok=True)

        k = uuid.uuid1()
        mfa_corpus_headname = f"mfa-{k}"
        audio_save_path = os.path.join(temp_dir, f"{mfa_corpus_headname}.wav")
        torchaudio.save(audio_save_path, audio['waveform'].squeeze(0), audio["sample_rate"])
        lab_save_path = os.path.join(temp_dir, f"{mfa_corpus_headname}.lab")
        with open(lab_save_path, "w", encoding="utf-8") as f:
            f.write(dubbing_draft)

        mm.unload_all_models()
        mm.soft_empty_cache()
        
        aligner = PretrainedAligner(
            corpus_directory=temp_dir,
            dictionary_path=DICTIONARY_PATH,
            acoustic_model_path=ACOUSTIC_MODEL_PATH,
            export_directory=out_temp_dir,
            clean=True
        )

        # 設定環境與資料庫 (MFA 3.x 初始化流程)
        aligner.setup()
        
        # 執行強制對齊
        aligner.align()
        
        # 匯出結果 (TextGrid)
        aligner.export_files(out_temp_dir)
        print("對齊完成，TextGrid 已匯出至:", out_temp_dir)
        print("Align Done, TextGrid Export To:", out_temp_dir)

        out_text_path = os.path.join(out_temp_dir, f"{mfa_corpus_headname}.TextGrid")
        tg = tgt.read_textgrid(out_text_path)
    
        # MFA 產生的 TextGrid 預設包含 'words' (字/詞) 與 'phones' (音素) 兩個層級
        words_tier = tg.get_tier_by_name('words')
        
        segments_list = []
        word_concat_list = []
        print("--- 字幕 (Words) 時間戳 (TimeStamp) ---")
        for interval in words_tier:
            # 過濾掉空白或靜音標記 (spn: 標音外詞, sil: 靜音)
            # if interval.text not in ['', 'spn', 'sil']:
            if interval.text not in ['spn', 'sil']:
                if show_verbose == True:
                    print(f"[{interval.start_time:.3f} - {interval.end_time:.3f}] {interval.text}")
                
                if interval.text == '' and ((interval.end_time-interval.start_time) > 0.5):
                    temp = {
                        "value": "",
                        "start": 0,
                        "end": 0
                    }
                    for word in word_concat_list:
                        if len(temp["value"]) + len(word["value"]) > segments_size:
                            segments_list.append(temp.copy())
                            temp["value"] = ""
                            temp["start"] = word["start"]
                        temp["value"] = temp["value"] + word["value"]
                        temp["end"] = word["end"]
                    segments_list.append(temp.copy())

                    word_concat_list.clear()
                
                item = {
                    "value": interval.text,
                    "start": interval.start_time, # 建議四捨五入到毫秒
                    "end": interval.end_time
                }
                word_concat_list.append(item)
        
        temp = {
            "value": "",
            "start": 0,
            "end": 0
        }
        for word in word_concat_list:
            if len(temp["value"]) + len(word["value"]) > segments_size:
                segments_list.append(temp.copy())
                temp["value"] = ""
                temp["start"] = word["start"]
            temp["value"] = temp["value"] + word["value"]
            temp["end"] = word["end"]
        segments_list.append(temp.copy())

        word_concat_list.clear()
        return (segments_list, )
                
            
