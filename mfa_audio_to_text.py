import os
from montreal_forced_aligner.alignment.pretrained import PretrainedAligner
# from montreal_forced_aligner.models import AcousticModel, DictionaryModel

def align_audio_to_text(corpus_dir, output_dir, acoustic_model, dictionary):
    """
    使用 MFA Python API 進行音訊與文字的強制對齊
    """
    # 初始化 Aligner
    # clean=True 可確保清理先前的暫存資料庫與特徵，避免狀態衝突
    aligner = PretrainedAligner(
        corpus_directory=corpus_dir,
        dictionary_path=dictionary,
        acoustic_model_path=acoustic_model,
        export_directory=output_dir,
        clean=True
    )

    try:
        # 設定環境與資料庫 (MFA 3.x 初始化流程)
        aligner.setup()
        
        # 執行強制對齊
        aligner.align()
        
        # 匯出結果 (TextGrid)
        aligner.export_files(output_dir)
        print("對齊完成，TextGrid 已匯出至:", output_dir)
        
    except Exception as e:
        print("對齊過程中發生錯誤:", e)

if __name__ == '__main__':
    # 因 MFA 底層使用 multiprocessing，需確保在 main 區塊內執行
    # CORPUS_DIR = "./dataset_folder"
    CORPUS_DIR = "dataset_folder"
    OUTPUT_DIR = "output_folder"
    
    # 若模型為本地檔案，可傳入絕對路徑；若是 MFA 內建模型，可直接傳入名稱
    ACOUSTIC_MODEL = "mandarin_mfa.zip" 
    DICTIONARY = "mandarin_mfa.dict"
    
    align_audio_to_text(CORPUS_DIR, OUTPUT_DIR, ACOUSTIC_MODEL, DICTIONARY)
    