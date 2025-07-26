FROM python:3.10

COPY ./requirements.txt /webapp/requirements.txt
COPY webapp/* /webapp

WORKDIR /webapp


RUN apt-get update && apt-get install -y wget
RUN wget https://huggingface.co/Mozilla/gemma-3-1b-it-llamafile/resolve/main/google_gemma-3-1b-it-Q6_K.llamafile
# Start Llamafile in the background, then start FastAPI
RUN pip install -r requirements.txt \
    && chmod +x google_gemma-3-1b-it-Q6_K.llamafile

CMD ./google_gemma-3-1b-it-Q6_K.llamafile & uvicorn main:app --host 0.0.0.0
