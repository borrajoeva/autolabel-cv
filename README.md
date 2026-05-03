# autolabel-cv
Pipeline de visión por computador para el etiquetado automático de datasets usando GroundingDINO.

## 1. entorno
python -m venv .venv

source .venv/bin/activate  
### En Windows: 
.venv\Scripts\activate

## 2. deps base
pip install --upgrade pip

pip install -r requirements.txt

## 3. GroundingDINO
git clone https://github.com/IDEA-Research/GroundingDINO.git

cd GroundingDINO

pip install --no-build-isolation -e .

cd ..

## 4. descargar modelo
mkdir models
### meter aquí groundingdino_swint_ogc.pth manualmente
https://huggingface.co/pengxian/grounding-dino/blob/main/groundingdino_swint_ogc.pth
