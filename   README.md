# House Segmentation Pipeline

This project now serves a Flask API for aerial house segmentation, with dataset preparation, training, evaluation, Docker support, and CI/CD.

## What changed

- Replaced the pretrained sentiment model with a segmentation pipeline based on DeepLabV3-ResNet50.
- Added secrets injection with `.env` support through `python-dotenv`.
- Added dataset preparation utilities for pixel mask generation and train/val/test splitting.
- Added training code that reports IoU and Dice score.
- Added GitHub Actions CI/CD to run tests, build the Docker image, and optionally push to Docker Hub.

## Main files

- [app.py](app.py)
- [config.py](config.py)
- [segmentation_utils.py](segmentation_utils.py)
- [dataset_tools.py](dataset_tools.py)
- [prepare_dataset.py](prepare_dataset.py)
- [train_segmentation.py](train_segmentation.py)
- [requirements.txt](requirements.txt)
- [Dockerfile](Dockerfile)
- [tests/test_app.py](tests/test_app.py)
- [.github/workflows/ci.yml](.github/workflows/ci.yml)

## Secrets injection

Copy [.env.example](.env.example) to `.env` and set the values you need.

Important keys:

- `SECRET_KEY`
- `API_TOKEN`
- `MODEL_CHECKPOINT`
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`
- `DOCKERHUB_IMAGE`

If `API_TOKEN` is set, the API expects `X-API-Token` on prediction requests.

## Dataset preparation

The dataset prep script supports either precomputed masks or polygon annotations.

Example:

```bash
python prepare_dataset.py \
  --images-dir data/raw/images \
  --annotations-file data/raw/annotations.json \
  --output-dir data/processed
```

Expected output structure:

- `data/processed/train/images`
- `data/processed/train/masks`
- `data/processed/val/images`
- `data/processed/val/masks`
- `data/processed/test/images`
- `data/processed/test/masks`

## Training

```bash
python train_segmentation.py \
  --data-dir data/processed \
  --output-dir outputs \
  --epochs 10 \
  --batch-size 4
```

Training outputs:

- `outputs/house_segmentation_best.pth`
- `outputs/history.json`
- `outputs/metrics.json`
- `outputs/training_curves.png`
- `outputs/sample_predictions.png`

## API usage

Health check:

```bash
curl http://localhost:5000/health
```

Prediction:

```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@sample_aerial.png"
```

If `API_TOKEN` is enabled:

```bash
curl -X POST http://localhost:5000/predict \
  -H "X-API-Token: $API_TOKEN" \
  -F "image=@sample_aerial.png"
```

## CI/CD

The GitHub Actions workflow does the following:

- Runs `pytest`
- Builds the Docker image
- Pushes to Docker Hub when the Docker Hub secrets are configured

Secrets to add in GitHub:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`
- `DOCKERHUB_IMAGE`

## Report deliverable

Use [REPORT.md](REPORT.md) as the source for the 4-page PDF report. It includes the structure for:

- Dataset description and preprocessing
- Model architecture and training approach
- IoU and Dice results
- Prediction visualizations
- Issues and mitigations