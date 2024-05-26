FROM python:3.9

WORKDIR /app

COPY multiplecolors.py /app/multiplecolors.py 
COPY aircraft_classifier_finale.h5 /app/aircraft_classifier_finale.h5
COPY color_classifier_finale.h5 /app/color_classifier_finale.h5

RUN pip install numpy tensorflow opencv-python-headless matplotlib scikit-learn pillow
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["python", "lumped.py"]
