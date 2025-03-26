import numpy as np
# Фикс для NumPy
if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'float': [np.float16, np.float32, np.float64],
        'complex': [np.complex64, np.complex128],
    }

import nemo.collections.asr as nemo_asr
from fastapi import File, UploadFile, FastAPI
from typing import List

app = FastAPI()

# Сохранение переданных в API файлов на диск 
def save_file(filename, data):
    with open(filename, 'wb') as f:
        f.write(data)

# Обрабатываем файлы. Читаем модель и передаем список файлов в модель для обработки.
def process_files(files):
    model = "QuartzNet15x5_golos_nemo.nemo"
    asr_model = nemo_asr.models.EncDecCTCModel.restore_from(model)
    return asr_model.transcribe(files, batch_size=20)

# Метод POST API принимает файлы для обработки
@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    audioFiles = []
    # Для каждого файла читаем данные и сохраняем на диск
    for file in files:
        contents = await file.read()
        save_file(file.filename, contents)
        audioFiles.append(file.filename)
    # Передаем список имен сохраненных файлов для обработки
    return {"text":process_files(audioFiles)}

# FastAPI пример (для проверки работы)
@app.get("/")
async def root():
    return {"message": "Hello World"}
    
@app.get("/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
