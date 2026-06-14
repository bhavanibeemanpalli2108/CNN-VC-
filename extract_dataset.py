import zipfile

with zipfile.ZipFile("vehicle-type-recognition.zip", "r") as zip_ref:
    zip_ref.extractall("data")