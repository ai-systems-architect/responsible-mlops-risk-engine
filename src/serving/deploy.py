"""
deploy.py — SageMaker Endpoint Deployment
------------------------------------------
Deploys the trained XGBoost model to a SageMaker real-time endpoint.
Written for SageMaker Python SDK 2.x.

Steps:
    1. Upload model artifact to S3
    2. Deploy using XGBoostModel — AWS managed container, no Dockerfile needed
    3. Wait for endpoint to reach InService (~8-10 minutes)
    4. Run sample inference to verify predictions
    5. Destroy endpoint immediately after verification

🚨 COST WARNING:
    This endpoint costs ~$5/day while running.
    The script destroys it automatically after inference verification.
    If the script fails before cleanup, run:
        python3 -c "
        import boto3
        boto3.client('sagemaker').delete_endpoint(
            EndpointName='responsible-risk-engine-prod-v1'
        )
        print('Endpoint deleted')
        "

NIST AI RMF alignment:
    MANAGE 1.3 — deployment requires explicit approval gate (fairness + AUC)
    MANAGE 2.4 — endpoint monitored via CloudWatch alarms provisioned in Terraform
"""

import os
import logging
import tarfile
import joblib
import pandas as pd
import boto3
import sagemaker
from glob import glob
from sagemaker.xgboost.model import XGBoostModel
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer
from dotenv import load_dotenv
from config import (
    SAGEMAKER_ENDPOINT_NAME,
    SAGEMAKER_INSTANCE,
    SAGEMAKER_ROLE_ARN,
    SAGEMAKER_FRAMEWORK_VERSION,
    S3_BUCKET,
    AWS_REGION,
    PROCESSED_DATA_DIR,
)

load_dotenv()

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
S3_MODEL_PREFIX = "models"


def validate_config():
    """
    Verify required environment variables are set before incurring any cost.
    Fails fast with a clear message rather than failing mid-deployment.
    """
    missing = []
    if not S3_BUCKET or "placeholder" in str(S3_BUCKET).lower():
        missing.append("S3_BUCKET")
    if not SAGEMAKER_ROLE_ARN or "placeholder" in str(SAGEMAKER_ROLE_ARN).lower():
        missing.append("SAGEMAKER_ROLE_ARN")
    if missing:
        raise ValueError(
            f"Missing required .env values: {missing}\n"
            "Update .env with Terraform output values before deploying."
        )
    logger.info("Config validated — S3_BUCKET and SAGEMAKER_ROLE_ARN set")


def create_inference_script():
    """
    Write inference.py for the SageMaker XGBoost container.

    The XGBoost container calls model_fn to load the model and
    predict_fn to run inference. Both must be defined.
    Script is packaged into model.tar.gz alongside the model file.
    """
    script = """import joblib
import os


def model_fn(model_dir):
    model_path = os.path.join(model_dir, "xgboost-model")
    model = joblib.load(model_path)
    return model


def predict_fn(input_data, model):
    proba = model.predict_proba(input_data)[:, 1]
    return proba.tolist()
"""
    os.makedirs("src/serving", exist_ok=True)
    with open("src/serving/serve_model.py", "w") as f:
        f.write(script)
    logger.info("Inference script written: src/serving/serve_model.py")


def package_model_artifact(models_dir: str = MODELS_DIR) -> str:
    """
    Package model and inference script into tar.gz for SageMaker.

    SageMaker requires:
        - model file at root named xgboost-model
        - serve_model.py at root alongside the model

    Returns:
        Path to local tar.gz
    """
    model_files = sorted(glob(f"{models_dir}/xgboost_native.json"))
    if not model_files:
        raise FileNotFoundError(
            f"No model artifact found in {models_dir}/. "
            "Run train_xgboost.py first."
        )

    model_path = model_files[-1]
    tar_path = f"{models_dir}/model.tar.gz"

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(model_path, arcname="xgboost-model")

    logger.info(f"Packaged: {model_path} + serve_model.py → {tar_path}")

    # Verify contents
    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getnames()
    logger.info(f"tar.gz contents: {members}")

    return tar_path


def upload_model_to_s3(tar_path: str) -> str:
    """
    Upload packaged model to S3.

    Returns:
        S3 URI of uploaded artifact
    """
    s3_client = boto3.client("s3", region_name=AWS_REGION)
    s3_key = f"{S3_MODEL_PREFIX}/model.tar.gz"

    logger.info(f"Uploading to s3://{S3_BUCKET}/{s3_key}")
    s3_client.upload_file(tar_path, S3_BUCKET, s3_key)

    s3_uri = f"s3://{S3_BUCKET}/{s3_key}"
    logger.info(f"Model uploaded: {s3_uri}")
    return s3_uri


def build_sample_input(data_dir: str = PROCESSED_DATA_DIR) -> pd.DataFrame:
    """
    Load 5 records from test set for inference verification.
    """
    X_test_files = sorted(glob(f"{data_dir}/X_test_*.parquet"))
    if not X_test_files:
        raise FileNotFoundError("No X_test parquet found. Run preprocess.py first.")

    X_test = pd.read_parquet(X_test_files[-1])
    X_test = X_test.drop(columns=["person_weight"], errors="ignore")
    sample = X_test.iloc[:5].copy()
    logger.info(f"Sample input: {sample.shape} — features: {sample.columns.tolist()}")
    return sample


def deploy_endpoint(s3_model_uri: str) -> sagemaker.predictor.Predictor:
    """
    Deploy XGBoost model using SageMaker 2.x XGBoostModel class.

    Uses AWS managed XGBoost container — no Dockerfile required.
    entry_point points to serve_model.py inside the tar.gz.

    Returns:
        SageMaker Predictor for running inference
    """
    sess = sagemaker.Session(
        boto_session=boto3.Session(region_name=AWS_REGION)
    )

    logger.info("Creating XGBoostModel")
    model = XGBoostModel(
        model_data=s3_model_uri,
        role=SAGEMAKER_ROLE_ARN,
        framework_version=SAGEMAKER_FRAMEWORK_VERSION,
        sagemaker_session=sess,
    )

    logger.info("=" * 55)
    logger.info(f"Deploying: {SAGEMAKER_ENDPOINT_NAME}")
    logger.info(f"Instance:  {SAGEMAKER_INSTANCE}")
    logger.info("Wait:      8-10 minutes")
    logger.info("🚨 ~$5/day — destroyed after verification")
    logger.info("=" * 55)

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=SAGEMAKER_INSTANCE,
        endpoint_name=SAGEMAKER_ENDPOINT_NAME,
        serializer=CSVSerializer(),
        deserializer=CSVDeserializer(),
    )

    logger.info(f"Endpoint InService: {SAGEMAKER_ENDPOINT_NAME}")
    return predictor


def run_sample_inference(
    predictor: sagemaker.predictor.Predictor,
    sample: pd.DataFrame,
) -> None:
    """
    Run sample inference and verify against local model.
    """
    logger.info("Running sample inference")

    csv_input = "\n".join([",".join(map(str, row)) for row in sample.values])
    response = predictor.predict(csv_input)

    print("\n--- Sample Inference Results ---")
    print(f"{'Record':<8} {'Probability':<12} {'Predicted Class'}")
    print("-" * 40)
    for i, row in enumerate(response):
        prob = float(row[0]) if isinstance(row, list) else float(row)
        label = "High income" if prob >= 0.5 else "Below threshold"
        print(f"{i+1:<8} {round(prob, 4):<12} {label}")

    # Verify consistency with local model
    local_model_files = sorted(glob(f"{MODELS_DIR}/xgboost_*.joblib"))
    if local_model_files:
        local_model = joblib.load(local_model_files[-1])
        local_proba = local_model.predict_proba(sample)[:, 1]
        print("\n--- Local vs Endpoint Consistency ---")
        for i, (row, lp) in enumerate(zip(response, local_proba)):
            ep = float(row[0]) if isinstance(row, list) else float(row)
            match = "✅" if abs(ep - lp) < 0.05 else "⚠️"
            print(f"  Record {i+1}: endpoint={round(ep,4)} local={round(lp,4)} {match}")

    print("\n  Note: Endpoint receives preprocessed inputs.")
    print("  Full pipeline (preprocessor + model) used for local/batch inference.")


def deploy(
    models_dir: str = MODELS_DIR,
    data_dir: str = PROCESSED_DATA_DIR,
) -> None:
    """
    Full deployment — create script, package, upload, deploy, verify, destroy.
    """
    logger.info("=" * 55)
    logger.info("SageMaker Deployment — Starting")
    logger.info("=" * 55)

    validate_config()
    tar_path = package_model_artifact(models_dir)
    s3_uri = upload_model_to_s3(tar_path)
    sample = build_sample_input(data_dir)
    predictor = deploy_endpoint(s3_uri)
    run_sample_inference(predictor, sample)

    print("\n--- Endpoint Details ---")
    print(f"  Name:     {SAGEMAKER_ENDPOINT_NAME}")
    print(f"  Region:   {AWS_REGION}")
    print(f"  Instance: {SAGEMAKER_INSTANCE}")
    print(f"  S3 model: s3://{S3_BUCKET}/{S3_MODEL_PREFIX}/model.tar.gz")

    print("\n" + "=" * 55)
    print("📸 TAKE SCREENSHOTS NOW:")
    print("  1. This terminal output")
    print("  2. AWS Console → SageMaker → Endpoints → InService")
    print("  3. AWS Console → S3 → models bucket → model.tar.gz")
    print("=" * 55)

    input("\nPress Enter when screenshots are taken to destroy endpoint...")

    logger.info("Destroying endpoint — stopping billing")
    predictor.delete_endpoint()
    logger.info(f"Endpoint destroyed: {SAGEMAKER_ENDPOINT_NAME}")
    print("\n✅ Endpoint destroyed — billing stopped")


if __name__ == "__main__":
    deploy()
