import os,sys
import yaml
from objectDetection.utils.main_utils import read_yaml_file
from objectDetection.logger import logging
from objectDetection.exception import AppException
from objectDetection.entity.config_entity import ModelTrainerConfig
from objectDetection.entity.artifacts_entity import ModelTrainerArtifact



class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.model_trainer_config = model_trainer_config


    

    def initiate_model_trainer(self,) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            logging.info("Unzipping data")
            # os.system("unzip /workspaces/fruit-vegetable-detection/artifacts/data_ingestion/data.zip")
            # os.system("rm /workspaces/fruit-vegetable-detection/artifacts/data_ingestion/data.zip")
            # Specify the source ZIP file path
            zip_file_path = "/workspaces/fruit-vegetable-detection/artifacts/data_ingestion/data.zip"

            # Specify the destination directory where you want to extract the contents
            destination_dir = "/workspaces/fruit-vegetable-detection/yolov9"

            # Create the destination directory if it doesn't exist
            os.makedirs(destination_dir, exist_ok=True)

            # Unzip the file into the destination directory
            os.system(f"unzip {zip_file_path} -d {destination_dir}")

            # Remove the original ZIP file after extraction
            os.remove(zip_file_path)

            with open("/workspaces/fruit-vegetable-detection/artifacts/data_ingestion/feature_store/finaldata/data1.yaml", 'r') as stream:
                num_classes = str(yaml.safe_load(stream)['nc'])

            model_config_file_name = self.model_trainer_config.weight_name.split(".")[0]
            print(model_config_file_name)

            config = read_yaml_file(f"yolov9/models/detect/{model_config_file_name}.yaml")

            config['nc'] = int(num_classes)


            with open(f'yolov9/models/detect/custom_{model_config_file_name}.yaml', 'w') as f:
                yaml.dump(config, f)

            print("#######################")

            # os.system(f"cd yolov9/ && python train_dual.py --workers 8 --device cpu --batch {self.model_trainer_config.batch_size} --data ..finaldata/data1.yaml --img 640 --cfg models/detect/custom_yolov9-c.yaml --weights '{self.model_trainer_config.no_epochs}' --name yolov9-c --hyp hyp.scratch-high.yaml --min-items 0 --epochs {self.model_trainer_config.no_epochs}")
            os.system(f"python yolov9/train.py \
            --batch {self.model_trainer_config.batch_size} --epochs {self.model_trainer_config.no_epochs} --img 640 --device cpu --min-items 0 --close-mosaic 15 \
            --data yolov9/finaldata/data1.yaml \
            --weights gelan-c.pt \
            --cfg yolov9/models/detect/gelan-c.yaml \
            --hyp yolov9/data/hyps/hyp.scratch-high.yaml")
            print("wgyufdscvdsavcj333")
            os.system("cp yolov9/runs/train/yolov9s_results/weights/best.pt yolov9/")
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            os.system(f"cp yolov9/runs/train/yolov9s_results/weights/best.pt {self.model_trainer_config.model_trainer_dir}/")
           
            os.system("rm -rf yolov9/runs")
            os.system("rm -rf yolov9/finaldata/train")
            os.system("rm -rf yolov9/finaldata/valid")
            os.system("rm -rf yolov9/finaldata/data1.yaml")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path="yolov9/best.pt",
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact


        except Exception as e:
            raise AppException(e, sys)