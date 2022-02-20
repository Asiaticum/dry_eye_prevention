# 概要

瞬きを検知して，一分あたりの瞬き回数が少ない時に注意喚起をする．



# OpenCVでリアルタイム撮影

```python
camera = cv2.VideoCapture(0)                # カメラCh.(ここでは0)を指定
 
# 撮影＝ループ中にフレームを1枚ずつ取得（qキーで撮影終了）
cv2.startWindowThread()
while True:
    ret, frame = camera.read()              # フレームを取得
    cv2.imshow('camera', frame)             # フレームを画面に表示
 
    # キー操作があればwhileループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# 撮影用オブジェクトとウィンドウの解放
camera.release()
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)
```


