FROM python
WORKDIR /app
COPY . .
COPY .streamlit /root/.streamlit
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["streamlit", "run", "app.py"]