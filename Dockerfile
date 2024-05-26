FROM python:3.9

WORKDIR /vlm_app

COPY locator2.py /app/locator2.py 

RUN pip install numpy tensorflow opencv-python-headless matplotlib scikit-learn pillow
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["python", "locator2.py"]
