FROM python:3.8-slim


WORKDIR /app 

COPY start_up.py ./start_script.py

COPY requirements.txt ./requirements.txt    

RUN pip install -r requirements.txt

EXPOSE 9000

CMD ["python", "start_script.py"]