FROM python3.13-slim

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY /src/ /app/src

ENTRYPOINT [ "python3","./src/go2_stand_up/go2_demo.py"]

