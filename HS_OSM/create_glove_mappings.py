import re
import json
import logging

# Configure logging for clear output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MappingGenerator:
    """
    Generates mappings for Out-of-Vocabulary (OOV) words by attempting to split them
    into constituent parts that exist in a given GloVe vocabulary.
    """
    def __init__(self, glove_path, missing_words_file, output_file="word_mappings.json"):
        self.glove_path = glove_path
        self.missing_words_file = missing_words_file
        self.output_file = output_file
        self.glove_vocab_sample = set()
        self.missing_words = []
        self.mappings = {}
        
    def load_glove_sample(self, sample_size=200000):
        """
        Loads a sample of the GloVe vocabulary for efficient existence checks.
        This avoids loading the entire multi-gigabyte file into memory.
        """
        logger.info(f"Loading GloVe vocabulary sample (up to {sample_size} words)...")
        count = 0
        
        try:
            with open(self.glove_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if count >= sample_size:
                        break
                    try:
                        word = line.split(' ', 1)[0]
                        self.glove_vocab_sample.add(word)
                        count += 1
                    except IndexError:
                        # Ignore malformed lines
                        continue
        except FileNotFoundError:
            logger.error(f"GloVe file not found at: {self.glove_path}")
            raise
                    
        logger.info(f"Loaded {len(self.glove_vocab_sample)} GloVe vocabulary samples.")
    
    def load_missing_words(self):
        """Loads the list of OOV words from a text file."""
        try:
            with open(self.missing_words_file, 'r', encoding='utf-8') as f:
                self.missing_words = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(self.missing_words)} missing words from {self.missing_words_file}.")
        except FileNotFoundError:
            logger.error(f"Missing words file not found at: {self.missing_words_file}")
            raise

    def word_exists_in_glove(self, word):
        """Checks if a word exists in the loaded GloVe vocabulary sample."""
        return word in self.glove_vocab_sample or word.lower() in self.glove_vocab_sample
    
    def create_mappings(self):
        """Iterates through missing words and attempts to generate a valid mapping for each."""
        logger.info("Starting to create mappings for OOV words...")
        
        for word in self.missing_words:
            # Attempt to find a valid split using a series of strategies
            mapped_parts = self.try_mapping_strategies(word)
            
            if mapped_parts:
                self.mappings[word] = mapped_parts
                logger.info(f"Created mapping for '{word}': {mapped_parts}")
        
        if self.missing_words:
            coverage = len(self.mappings) / len(self.missing_words) * 100
            logger.info(f"Successfully created {len(self.mappings)} mappings ({coverage:.1f}% coverage of missing words).")
    
    def try_mapping_strategies(self, word):
        """
        Applies a series of strategies to find a valid split for a word.
        The first strategy that succeeds is returned.
        """
        # Strategy 1: Split by hyphen
        if '-' in word:
            parts = word.split('-')
            if all(self.word_exists_in_glove(part) for part in parts):
                return parts
        
        # Strategy 2: Split by common compound word patterns (e.g., "meterbox" -> "meter", "box")
        compound_patterns = {
            r'^(.+?)(box|machine|fan|board|case|holder|net|cart|bin|room|truck|controller|door|booth|glass|set)$': 
                lambda m: [m.group(1), m.group(2)],
            r'^(baby|bird|fish|air|gas|electric|water|video|fire|glass|poker|stick|paper|hard|pot|knife)(.+)$': 
                lambda m: [m.group(1), m.group(2)]
        }
        
        for pattern, splitter in compound_patterns.items():
            match = re.match(pattern, word.lower())
            if match:
                parts = splitter(match)
                if all(self.word_exists_in_glove(part) for part in parts):
                    return parts
        
        # Strategy 3: Split by camelCase (e.g., "babyCarriage" -> "baby", "carriage")
        camel_case_parts = re.findall(r'[A-Z][a-z]*|[a-z]+', word)
        if len(camel_case_parts) > 1:
            # Convert parts to lowercase for checking
            lower_parts = [p.lower() for p in camel_case_parts]
            if all(self.word_exists_in_glove(part) for part in lower_parts):
                return lower_parts
        
        # Strategy 4: Use a hardcoded dictionary of common splits
        # This is useful for specific, known compound words in datasets like Pascal-Context 459.
        common_splits = {
            "babycarriage": ["baby", "carriage"], "baseballbat": ["baseball", "bat"],
            "basketballbackboard": ["basketball", "backboard"], "bottleopener": ["bottle", "opener"],
            "cabinetdoor": ["cabinet", "door"], "cameralens": ["camera", "lens"],
            "casetterecorder": ["casette", "recorder"], "cashregister": ["cash", "register"],
            "clothestree": ["clothes", "tree"], "controlbooth": ["control", "booth"],
            "copyingmachine": ["copying", "machine"], "cuttingboard": ["cutting", "board"],
            "disccase": ["disc", "case"], "drinkdispenser": ["drink", "dispenser"],
            "drinkingmachine": ["drinking", "machine"], "electricfan": ["electric", "fan"],
            "electriciron": ["electric", "iron"], "electricpot": ["electric", "pot"],
            "electricsaw": ["electric", "saw"], "electronickeyboard": ["electronic", "keyboard"],
            "fireextinguisher": ["fire", "extinguisher"], "firehydrant": ["fire", "hydrant"],
            "fishingnet": ["fishing", "net"], "fishingpole": ["fishing", "pole"],
            "gamecontroller": ["game", "controller"], "gamemachine": ["game", "machine"],
            "gascylinder": ["gas", "cylinder"], "gashood": ["gas", "hood"],
            "gasstove": ["gas", "stove"], "glassmarble": ["glass", "marble"],
            "harddiskdrive": ["hard", "disk", "drive"], "ironingboard": ["ironing", "board"],
            "kitchenrange": ["kitchen", "range"], "knifeblock": ["knife", "block"],
            "laddertruck": ["ladder", "truck"], "meterbox": ["meter", "box"],
            "musicalinstrument": ["musical", "instrument"], "oxygenbottle": ["oxygen", "bottle"],
            "paperbox": ["paper", "box"], "papercutter": ["paper", "cutter"],
            "pencontainer": ["pen", "container"], "pokerchip": ["poker", "chip"],
            "pottedplant": ["potted", "plant"], "signallight": ["signal", "light"],
            "spicecontainer": ["spice", "container"], "stickynote": ["sticky", "note"],
            "surveillancecamera": ["surveillance", "camera"], "swimring": ["swim", "ring"],
            "telephonebooth": ["telephone", "booth"], "toycar": ["toy", "car"],
            "tvmonitor": ["tv", "monitor"], "vacuumcleaner": ["vacuum", "cleaner"],
            "vendingmachine": ["vending", "machine"], "videogameconsole": ["video", "game", "console"],
            "waterdispenser": ["water", "dispenser"], "waterskateboard": ["water", "skateboard"]
        }
        
        if word.lower() in common_splits:
            parts = common_splits[word.lower()]
            if all(self.word_exists_in_glove(part) for part in parts):
                return parts
        
        # Strategy 5: Recursive binary splitting
        # This is a brute-force attempt to find any valid two-part split.
        for i in range(3, len(word) - 2):
            part1 = word[:i].lower()
            part2 = word[i:].lower()
            
            if self.word_exists_in_glove(part1) and self.word_exists_in_glove(part2):
                return [part1, part2]
                
        return None
    
    def save_mappings(self):
        """Saves the generated mappings to a JSON file."""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.mappings, f, indent=2, sort_keys=True)
        logger.info(f"Mappings have been saved to: {self.output_file}")
    
    def run(self):
        """Executes the entire mapping generation pipeline."""
        self.load_glove_sample()
        self.load_missing_words()
        self.create_mappings()
        self.save_mappings()
        return self.mappings

if __name__ == "__main__":
    # Example usage: Generate mappings for missing WordNet words from the PC-459 dataset
    generator = MappingGenerator(
        glove_path="HS_OSM/glove.840B.300d.txt",  # Path to your GloVe file
        missing_words_file="HS_OSM/word_missing_mapping/pc459_wordnet_missing.txt",
        output_file="HS_OSM/word_missing_mapping/pc459_wordnet_mappings.json"
    )
    mappings = generator.run()
    
    # Report any words that could not be mapped
    unmapped_words = [w for w in generator.missing_words if w not in mappings]
    if unmapped_words:
        print(f"\nCould not find a mapping for {len(unmapped_words)} words:")
        for word in unmapped_words[:20]:  # Display the first 20 unmapped words
            print(f"- {word}")
        if len(unmapped_words) > 20:
            print(f"... and {len(unmapped_words) - 20} more.")
    else:
        print("\nAll missing words were successfully mapped.")
