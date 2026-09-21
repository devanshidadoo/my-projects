import os
import sys
import base64
import requests
import tempfile
from typing import Dict, Any, Optional, Tuple, Type
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from pydantic import BaseModel, Field
from urllib.parse import urlparse

# Add the parent directory (project root) to the Python path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

from agents.config import Config as config

class PestDetectionInput(BaseModel):
    """Input schema for the pest detection tool."""
    image: str = Field(description="Public URL of the image for pest detection")

class PestDetectionTool(BaseTool):
    """
    LangChain tool for pest detection that combines a Gradio API endpoint 
    with OpenAI for comprehensive pest analysis and remedy recommendation.
    """
    
    name: str = "PestDetectionTool"
    description: str = """
    Detects pests in images using a machine learning model and provides remedies.
    Takes a public image URL as input.
    """
    args_schema: Type[BaseModel] = PestDetectionInput
    
    # Define the attributes as class fields
    llm: Any = Field(default=None, exclude=True)
    pest_classes: Dict[int, str] = Field(default_factory=dict, exclude=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize the LLM
        self.llm = ChatOpenAI(
            api_key = config.OPENAI_API_KEY,
            model = "gpt-4o", 
        )
        
        # Pest class mapping from the provided classes.txt
        self.pest_classes = {
            1: "rice leaf roller",
            2: "rice leaf caterpillar",
            3: "paddy stem maggot",
            4: "asiatic rice borer",
            5: "yellow rice borer",
            6: "rice gall midge",
            7: "Rice Stemfly",
            8: "brown plant hopper",
            9: "white backed plant hopper",
            10: "small brown plant hopper",
            11: "rice water weevil",
            12: "rice leafhopper",
            13: "grain spreader thrips",
            14: "rice shell pest",
            15: "grub",
            16: "mole cricket",
            17: "wireworm",
            18: "white margined moth",
            19: "black cutworm",
            20: "large cutworm",
            21: "yellow cutworm",
            22: "red spider",
            23: "corn borer",
            24: "army worm",
            25: "aphids",
            26: "Potosiabre vitarsis",
            27: "peach borer",
            28: "english grain aphid",
            29: "green bug",
            30: "bird cherry-oataphid",
            31: "wheat blossom midge",
            32: "penthaleus major",
            33: "longlegged spider mite",
            34: "wheat phloeothrips",
            35: "wheat sawfly",
            36: "cerodonta denticornis",
            37: "beet fly",
            38: "flea beetle",
            39: "cabbage army worm",
            40: "beet army worm",
            41: "Beet spot flies",
            42: "meadow moth",
            43: "beet weevil",
            44: "sericaorient alismots chulsky",
            45: "alfalfa weevil",
            46: "flax budworm",
            47: "alfalfa plant bug",
            48: "tarnished plant bug",
            49: "Locustoidea",
            50: "lytta polita",
            51: "legume blister beetle",
            52: "blister beetle",
            53: "therioaphis maculata Buckton",
            54: "odontothrips loti",
            55: "Thrips",
            56: "alfalfa seed chalcid",
            57: "Pieris canidia",
            58: "Apolygus lucorum",
            59: "Limacodidae",
            60: "Viteus vitifoliae",
            61: "Colomerus vitis",
            62: "Brevipoalpus lewisi McGregor",
            63: "oides decempunctata",
            64: "Polyphagotars onemus latus",
            65: "Pseudococcus comstocki Kuwana",
            66: "parathrene regalis",
            67: "Ampelophaga",
            68: "Lycorma delicatula",
            69: "Xylotrechus",
            70: "Cicadella viridis",
            71: "Miridae",
            72: "Trialeurodes vaporariorum",
            73: "Erythroneura apicalis",
            74: "Papilio xuthus",
            75: "Panonchus citri McGregor",
            76: "Phyllocoptes oleiverus ashmead",
            77: "Icerya purchasi Maskell",
            78: "Unaspis yanonensis",
            79: "Ceroplastes rubens",
            80: "Chrysomphalus aonidum",
            81: "Parlatoria zizyphus Lucus",
            82: "Nipaecoccus vastalor",
            83: "Aleurocanthus spiniferus",
            84: "Tetradacus c Bactrocera minax",
            85: "Dacus dorsalis(Hendel)",
            86: "Bactrocera tsuneonis",
            87: "Prodenia litura",
            88: "Adristyrannus",
            89: "Phyllocnistis citrella Stainton",
            90: "Toxoptera citricidus",
            91: "Toxoptera aurantii",
            92: "Aphis citricola Vander Goot",
            93: "Scirtothrips dorsalis Hood",
            94: "Dasineura sp",
            95: "Lawana imitata Melichar",
            96: "Salurnis marginella Guerr",
            97: "Deporaus marginatus Pascoe",
            98: "Chlumetia transversa",
            99: "Mango flat beak leafhopper",
            100: "Rhytidodera bowrinii white",
            101: "Sternochetus frigidus",
            102: "Cicadellidae"
        }
    
    def is_valid_url(self, url: str) -> bool:
        """Validate if the provided string is a valid URL."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def download_image_from_url(self, url: str) -> Optional[str]:
        """
        Download image from URL and save to temporary file.
        Returns:
            str: Path to the downloaded temporary file, or None if failed
        """
        try:
            if not self.is_valid_url(url):
                raise ValueError("Invalid URL provided")
            
            # Set headers to mimic a browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()
            
            # Check if the content is an image
            content_type = response.headers.get('content-type', '').lower()
            if not content_type.startswith('image/'):
                raise ValueError(f"URL does not point to an image. Content-Type: {content_type}")
            
            # Create temporary file with appropriate extension
            file_extension = '.jpg'  # Default extension
            if 'png' in content_type:
                file_extension = '.png'
            elif 'gif' in content_type:
                file_extension = '.gif'
            elif 'webp' in content_type:
                file_extension = '.webp'
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
            
            # Download and save the image
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
            
            temp_file.close()
            return temp_file.name
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading image from URL: {e}")
            return None
        except Exception as e:
            print(f"Error processing image URL: {e}")
            return None
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64 for API calls."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def call_gradio_api(self, image_path: str) -> Tuple[Optional[int], Optional[float]]:
        """
        Call the Hugging Face Gradio API endpoint for pest classification.
        Args:
            image_path: Path to the downloaded image file
        Returns:
            Tuple[int, float]: (predicted_class, confidence_probability)
        """
        if not os.path.exists(image_path):
            return None, None
        try:
            from gradio_client import Client, handle_file
            client = Client("subarnoM/EffecientNetPest")
            result = client.predict(
                image=handle_file(image_path),
                api_name="/predict"
            )
            if isinstance(result, (list, tuple)) and len(result) >= 1:
                prediction_data = result[0]
                if isinstance(prediction_data, dict) and 'label' in prediction_data:
                    predicted_label = prediction_data['label']
                    confidences_list = prediction_data.get('confidences', [])
                    # Find the confidence for the predicted label
                    conf = None
                    if confidences_list:
                        for conf_item in confidences_list:
                            if conf_item.get('label', '') == predicted_label:
                                conf = conf_item.get('confidence', 0.0)
                                break
                        if conf is None:
                            conf = confidences_list[0].get('confidence', 0.0)
                    else:
                        conf = 0.0
                    # Extract class number from label
                    class_num_str = predicted_label.replace('Class_', '')
                    try:
                        predicted_class = int(class_num_str) + 1
                    except Exception:
                        predicted_class = None
                    return predicted_class, conf
        except Exception:
            return None, None

    def analyze_with_openai_vision(self, image_path: str, pest_name: Optional[str] = None) -> str:
        """
        Analyze image with OpenAI GPT-4 Vision for pest detection and remedy.
        Args:
            image_path: Path to the image
            pest_name: If provided, ask for remedy for specific pest
        Returns:
            str: Analysis result and remedy recommendation
        """
        try:
            base64_image = self.encode_image_to_base64(image_path)
            if pest_name:
                prompt = f"""
                I have detected a pest in this image identified as: {pest_name}
                
                Please provide:
                1. Comprehensive treatment and control measures (Steps)
                2. Prevention strategies
                3. Expected timeline for treatment effectiveness
                
                Please be specific and practical in your recommendations.
                """
            else:
                prompt = """
                Please analyze this image and determine:
                
                1. Is there a pest visible in this image? (Yes/No)
                2. If yes, what type of pest is it? Please be as specific as possible.
                3. If it's a pest, provide:
                   - Comprehensive treatment and control measures
                   - Prevention strategies
                   - Expected timeline for treatment effectiveness
                4. If it's not a pest, say exactly "THE ENTERED IMAGE IS NOT A PEST, PLEASE ENTER IMAGE OF A PEST".
                
                Please be thorough and practical in your analysis and recommendations.
                """
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            )
            response = self.llm.invoke([message])
            return response.content
        except Exception as e:
            return f"Error analyzing image with OpenAI: {str(e)}"
    
    def _run(self, image: str) -> str:
        """
        Main execution method for the tool.
        Args:
            image: Public URL of the image
        Returns:
            str: Complete analysis and remedy recommendation
        """
        temp_image_path = None
        try:
            # Download image from URL
            temp_image_path = self.download_image_from_url(image)
            if temp_image_path is None:
                return "Error: Failed to download image from the provided URL. Please ensure the URL is valid and points to an image."
            
            # Proceed with pest detection using the downloaded image
            predicted_class, confidence = self.call_gradio_api(temp_image_path)
            output_lines = []
            output_lines.append(f"Predicted Class: {predicted_class}")
            if confidence is not None:
                output_lines.append(f"Confidence: {confidence:.2%}")
            else:
                output_lines.append("Confidence: N/A")
            
            if confidence is not None and predicted_class is not None and confidence >= 0.75:
                pest_name = self.pest_classes.get(predicted_class, f"Unknown pest (Class {predicted_class})")
                openai_analysis = self.analyze_with_openai_vision(temp_image_path, pest_name)
                output_lines.append(openai_analysis)
            else:
                openai_analysis = self.analyze_with_openai_vision(temp_image_path)
                output_lines.append(openai_analysis)
            
            return "\n".join(output_lines)
            
        except Exception as e:
            return f"Error in pest detection: {str(e)}"
        finally:
            # Clean up temporary file
            if temp_image_path and os.path.exists(temp_image_path):
                try:
                    os.unlink(temp_image_path)
                except Exception:
                    pass  # Ignore cleanup errors
    
    async def _arun(self, image: str) -> str:
        """
        Asynchronous version of the tool.
        Args:
            image: Public URL of the image
        Returns:
            str: Complete analysis and remedy recommendation
        """
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return await loop.run_in_executor(
            None, self._run, image
        )   

if __name__ == "__main__":
    import asyncio

    IMAGE_URL = input("Enter image URL: ").strip()
    if IMAGE_URL:
        pest_tool = PestDetectionTool()
        try:    
            result = asyncio.run(pest_tool._arun(IMAGE_URL))
            print(result)
        except Exception as e:
            print(f"❌ Error running pest detection: {e}")
    else:
        print("❌ Missing image URL")