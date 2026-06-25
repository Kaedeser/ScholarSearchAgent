FROM mzx/deberta-train:2.2.2-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

RUN python -m pip install --no-cache-dir --force-reinstall \
    "numpy==1.26.4" \
    "transformers==4.57.6"

WORKDIR /app
COPY app.py /app/app.py

EXPOSE 8000
CMD ["python", "/app/app.py"]
