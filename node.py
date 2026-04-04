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
from datetime import datetime


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
                "segments_fill_space_seconds": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.1,
                    "tooltip": "When the breathing interval is greater than this value but less than the cutoff decision interval, add a space at the end of the word to improve the readability of the subtitles."
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
                "FILTER_CHAR_spn": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Filter out 'Spoken Noise' (spn) tags. MFA uses 'spn' for coughs, breaths, or unrecognizable sounds. Enabling this prevents noise from interfering with text alignment."
                }),
                "FILTER_CHAR_sil": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Filter out 'Silence' (sil) tags. Represents pauses or environment noise. While 'sil' is useful for segmentation, filtering it out ensures the text-to-audio mapping remains focused on actual speech."
                }),
                "FILTER_CHAR_unk": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Filter out 'Unknown' (unk) tags. These occur for OOV (Out-of-Vocabulary) words or unclear speech. Enabling this allows the ASR Mapper to use DP algorithms to correctly fill in original text at these gaps."
                })
            },
            "optional": {
                "show_verbose": ("BOOLEAN", {
                    "default": False
                }),
                "export_verbose_log": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "export MFA TextGrid Context"
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
    
    def audioToString(self, audio, dubbing_draft, ACOUSTIC_MODEL_PATH, DICTIONARY_PATH, segments_merge_and_cutoff_seconds=0.5, segments_fill_space_seconds=0.2, enable_auto_split_segments=True, segments_size=25, FILTER_CHAR_spn=True, FILTER_CHAR_sil=True, FILTER_CHAR_unk=True, show_verbose=False, export_verbose_log=False, show_system_info=False, MFA_quiet_mode=True, MFA_verbose_mode=False, MFA_clean_lock=True, MFA_beam=100, MFA_retry_beam=400, unique_id=0):
        
        # segments_merge_and_cutoff_seconds
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

        # segments_fill_space_seconds
        if isinstance(segments_fill_space_seconds, float) == True and math.isnan(segments_fill_space_seconds) == True:
            segments_fill_space_seconds = 0.2
        elif isinstance(segments_fill_space_seconds, float) == False:
            try:
                segments_fill_space_seconds = float(segments_fill_space_seconds)
            except:
                segments_fill_space_seconds = 0.2
            
            if math.isnan(segments_fill_space_seconds) == True:
                segments_fill_space_seconds = 0.2
        elif segments_fill_space_seconds < 0.1:
            segments_fill_space_seconds = 0.2

        output_dir_root = folder_paths.get_output_directory()
        output_dir = os.path.join(output_dir_root, "ComfyUI-MontrealForcedAligner")
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_save_path = os.path.join(output_dir, f"{ts}.log")
        

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


        filter_word_list = []

        if FILTER_CHAR_spn == True:
            filter_word_list.append("spn")
        if FILTER_CHAR_sil == True:
            filter_word_list.append("sil")
        if FILTER_CHAR_unk == True:
            filter_word_list.append("<unk>")

        log_fd = None
        if export_verbose_log == True:
            try:
                log_fd = open(log_save_path, "w", encoding="utf-8")
            except Exception as e:
                print(e)

        last_word_time = words_tier[0].start_time if len(words_tier) > 0 else 0

        try:

            
            for interval in words_tier:
                # 過濾掉空白或靜音標記 (spn: 標音外詞, sil: 靜音)
                # if interval.text not in ['', 'spn', 'sil']:
                # if interval.text not in filter_word_list:
                log_ln = f"[{interval.start_time:.3f} - {interval.end_time:.3f}] {interval.text}\n"
                if show_verbose == True:
                    print(log_ln, end="")
                if export_verbose_log == True and log_fd:
                    log_fd.write(log_ln)
                
                    
                
                # filter '\n'
                interval.text = interval.text.replace("\n", "")
                interval_diff_time = interval.start_time - last_word_time
                last_word_time = interval.start_time

                if (interval_diff_time < segments_merge_and_cutoff_seconds) and (interval_diff_time > segments_fill_space_seconds):
                    interval.text = f"{interval.text} "
                elif (interval_diff_time >= segments_merge_and_cutoff_seconds):
                    temp = {
                        "value": "",
                        "start": word_concat_list[0]["start"] if len(word_concat_list) > 0 else 0,
                        "end": 0
                    }

                    if enable_auto_split_segments == False:

                        for word in word_concat_list:
                            temp["value"] = temp["value"] + word["value"]
                            temp["end"] = word["end"]
                        segments_list.append(temp.copy())

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
                # =============if end line========================
                
                item = {
                    "value": interval.text if interval.text not in filter_word_list else "",
                    "start": interval.start_time,
                    "end": interval.end_time
                }
                word_concat_list.append(item)
        finally:
            if log_fd:
                log_fd.close()
        
        # It has poor performance, but this is for the sake of easy-to-read source code structure.
        if len(word_concat_list) > 0:
            temp = {
                "value": "",
                "start": word_concat_list[0]["start"] if len(word_concat_list) > 0 else 0,
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
                
            
