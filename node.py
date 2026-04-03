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
                    "tooltip": "TTS generates audio based on the original text."
                }),
                "dubbing_draft": ("STRING", {
                    "forceInput": True,
                    "tooltip": "The original text provided to the TTS model for reading aloud"
                }),
                "ACOUSTIC_MODEL_PATH": ("STRING", {
                    "tooltip": "MFA ACOUSTIC MODEL PATH"
                }),
                "DICTIONARY_PATH": ("STRING", {
                    "tooltip": "MFA DICTIONARY PATH"
                }),
                "segments_merge_and_cutoff_seconds": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "tooltip": "Unit: (s), The alignment provided by MFA is based on each individual word. Therefore, when generating subtitles, merging and segmenting are required. The basis for merging and segmenting mainly occurs during breaths in speech. This parameter determines the interval between breaths at which segmentation is necessary."
                }),
                "enable_auto_split_segments": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If a single field is too long, should it be segmented? If not, the subtitle for a single sentence might be very long."
                }),
                "segments_size": ("INT", {
                    "min": 5,
                    "default": 25,
                    "tooltip": "When performing field splitting, the maximum number of characters in each field is limited."
                }),
            },
            "optional": {
                "show_verbose": ("BOOLEAN", {
                    "default": False
                }),
                "show_system_info": ("BOOLEAN", {
                    "default": False
                }),
                "MFA_quiet_mode": ("BOOLEAN", {
                    "default": True
                }),
                "MFA_verbose_mode": ("BOOLEAN", {
                    "default": False
                }),
                "MFA_clean_lock": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "It is recommended to enable this feature; otherwise, the behavior of the MFA may become unpredictable. \n Avoid dirty data pollution / Resolve database deadlock issues (MFA uses SQLite or PostgreSQL to record state. If the previous execution was abnormally interrupted due to an error, the database may leave behind a bunch of error markers.)\n -True: Before initiating the alignment task, MFA deletes all old temporary archives and database records corresponding to the corpus. This ensures that each execution is a completely new state 'starting from scratch'. \n - False: MFA will attempt to reuse previous caches. If the audio file and text remain unchanged, it will skip time-consuming steps such as feature extraction."
                }),
                "MFA_beam": ("INT", {
                    "min": 10,
                    "default": 100,
                    "tooltip": "The beam defines the number of 'probabilistic paths' that the algorithm retains in each frame of audio under normal alignment conditions. When the model is aligning, it calculates multiple possible paths. The beam sets a threshold, and only paths with a sufficiently high probability are retained.\n - Larger beams: offer more search paths and more accurate alignment, but consume more memory and CPU.\n - Smaller Beams: Search speed is extremely fast, but if the audio quality is poor or the speech rate is too fast, the model may 'break down' (throw an alignment failure) because it cannot find a good enough path.\n - Preset value: usually between 10 and 100 (depending on the model version)."
                }),
                "MFA_retry_beam": ("INT", {
                    "min": 40,
                    "default": 400,
                    "tooltip": "When a normal beam cannot find a complete alignment path (usually resulting in an 'Alignment failed' error), the MFA will not give up directly but will initiate a retry mechanism, which is where the retry_beam comes in.\n - Functionality: This is a 'backup plan'. If the initial alignment fails, the MFA will rescan the audio using a larger width set by retry_beam.\n Impact: This is usually set much larger than beam (e.g., the default might be 400 or higher), and it can salvage segments that fail to align on the first attempt due to heavy background noise, extremely unclear pronunciation, or irregular speaking speed.\n - Cost: Entering the Retry phase will significantly increase the processing time of the audio segment."
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
    
    def audioToString(self, audio, dubbing_draft, ACOUSTIC_MODEL_PATH, DICTIONARY_PATH, segments_merge_and_cutoff_seconds=0.5, enable_auto_split_segments=True, segments_size=25, show_verbose=False, show_system_info=False, MFA_quiet_mode=True, MFA_verbose_mode=False, MFA_clean_lock=True, MFA_beam=100, MFA_retry_beam=400, unique_id=0):
        

        if isinstance(segments_merge_and_cutoff_seconds, float) == True and math.isnan(segments_merge_and_cutoff_seconds) == True:
            segments_merge_and_cutoff_seconds = 0.5
        elif isinstance(segments_merge_and_cutoff_seconds, float) == False:
            try:
                segments_merge_and_cutoff_seconds = float(segments_merge_and_cutoff_seconds)
            except:
                segments_merge_and_cutoff_seconds = 0.5
            
            if math.isnan(segments_merge_and_cutoff_seconds) == True:
                segments_merge_and_cutoff_seconds = 0.5
        elif segments_merge_and_cutoff_seconds < 0.1:
            segments_merge_and_cutoff_seconds = 0.5
        

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

        if show_system_info == True:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                print("========> BEFORE RELEASE:")
                print("device:", device)
                print("name:", torch.cuda.get_device_name(device))
                print("allocated GB:", torch.cuda.memory_allocated(device) / 1024**3)
                print("reserved GB:", torch.cuda.memory_reserved(device) / 1024**3)

        mm.unload_all_models()
        mm.soft_empty_cache()

        if show_system_info == True:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                print("========> AFTER RELEASE:")
                print("device:", device)
                print("name:", torch.cuda.get_device_name(device))
                print("allocated GB:", torch.cuda.memory_allocated(device) / 1024**3)
                print("reserved GB:", torch.cuda.memory_reserved(device) / 1024**3)

        aligner = None
        try:
            aligner = PretrainedAligner(
                corpus_directory=temp_dir,
                dictionary_path=DICTIONARY_PATH,
                acoustic_model_path=ACOUSTIC_MODEL_PATH,
                export_directory=out_temp_dir,
                clean=MFA_clean_lock,
                quiet=MFA_quiet_mode,
                verbose=MFA_verbose_mode,
                beam=MFA_beam,
                retry_beam=MFA_retry_beam
            )
            # 設定環境與資料庫 (MFA 3.x 初始化流程)
            aligner.setup()
            
            # 執行強制對齊
            aligner.align()
            
            # 匯出結果 (TextGrid)
            aligner.export_files(out_temp_dir)
            print("Align Done, TextGrid Export To:", out_temp_dir)
        except Exception as e:
            print(f"MFA Execute ERROR ================>: {e}")
            raise e
        finally:
        # 這是解決 WinError 32 的關鍵
            if aligner is not None:
                try:
                    # 關閉資料庫連接並清理臨時檔案
                    aligner.cleanup() 
                    # 徹底銷毀物件
                    del aligner 
                except:
                    pass

        

        out_text_path = os.path.join(out_temp_dir, f"{mfa_corpus_headname}.TextGrid")
        tg = tgt.read_textgrid(out_text_path)
    
        # MFA 產生的 TextGrid 預設包含 'words' (字/詞) 與 'phones' (音素) 兩個層級
        words_tier = tg.get_tier_by_name('words')
        
        segments_list = []
        word_concat_list = []
        print("--- Words Timestamp ---")
        for interval in words_tier:
            # 過濾掉空白或靜音標記 (spn: 標音外詞, sil: 靜音)
            # if interval.text not in ['', 'spn', 'sil']:
            if interval.text not in ['spn', 'sil', 'unk']:
                if show_verbose == True:
                    print(f"[{interval.start_time:.3f} - {interval.end_time:.3f}] {interval.text}")
                
                if interval.text == '' and ((interval.end_time-interval.start_time) > segments_merge_and_cutoff_seconds):
                    temp = {
                        "value": "",
                        "start": 0,
                        "end": 0
                    }

                    if enable_auto_split_segments == False:
                        temp["start"] = word_concat_list[0]["start"] if len(word_concat_list) > 0 else 0
                        for word in word_concat_list:
                            temp["value"] = temp["value"] + word["value"]
                            temp["end"] = word["end"]
                    else:
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

        if os.path.exists(out_text_path):
            os.remove(out_text_path)
        if os.path.exists(lab_save_path):
            os.remove(lab_save_path)
        if os.path.exists(audio_save_path):
            os.remove(audio_save_path)

        return (segments_list, )
                
            
