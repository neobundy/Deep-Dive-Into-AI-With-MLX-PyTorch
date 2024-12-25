# Testing the Whispering MLX Chatbot Ideation

- Follow the instructions in the README.md first. That'll get you up and running just fine.
- Run `audio_server.py`
- Run streamlit app `streamlit run main.py`
- Click `Use Audio` and `Record` and say something. When finished click `Stop`. You will get your transcription in markdown box in the main area.


## Llama CPP

```python
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
```
