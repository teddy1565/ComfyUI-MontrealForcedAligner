import tgt
import os

def extract_timestamps(textgrid_path):
    """
    解析 TextGrid 檔案以取得字幕與讀音的時間戳
    """
    # 讀取 TextGrid
    tg = tgt.read_textgrid(textgrid_path)
    
    # MFA 產生的 TextGrid 預設包含 'words' (字/詞) 與 'phones' (音素) 兩個層級
    words_tier = tg.get_tier_by_name('words')
    phones_tier = tg.get_tier_by_name('phones')
    
    print("--- 字幕 (Words) 時間戳 ---")
    for interval in words_tier:
        # 過濾掉空白或靜音標記 (spn: 標音外詞, sil: 靜音)
        # if interval.text not in ['', 'spn', 'sil']:
        if interval.text not in ['', 'spn', 'sil']:
            print(f"[{interval.start_time:.3f} - {interval.end_time:.3f}] {interval.text}")

    # print("\n--- 讀音 (Phones) 時間戳 ---")
    # for interval in phones_tier:
    #     if interval.text not in ['', 'spn', 'sil']:
    #         print(f"[{interval.start_time:.3f} - {interval.end_time:.3f}] {interval.text}")

if __name__ == '__main__':
    target_textgrid = "./output_folder/test02.TextGrid"
    if os.path.exists(target_textgrid):
        extract_timestamps(target_textgrid)