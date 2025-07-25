FROM python:3.10

COPY ./requirements.txt /webapp/requirements.txt
COPY webapp/* /webapp
COPY google_gemma-3-1b-it-Q6_K.llamafile /webapp/google_gemma-3-1b-it-Q6_K.llamafile

WORKDIR /webapp

RUN pip install -r requirements.txt \
    && chmod +x google_gemma-3-1b-it-Q6_K.llamafile

# Start Llamafile in the background, then start FastAPI
CMD ./google_gemma-3-1b-it-Q6_K.llamafile & uvicorn main:app --host 0.0.0.0