#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import logging
from nltk.corpus import wordnet as wn

# Configure logging for clear output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WordNetMappingGenerator:
    """
    Generates mappings for class names that are Out-of-Vocabulary (OOV) in WordNet.
    It attempts to split compound words into constituent parts that exist in WordNet.
    """
    
    def __init__(self, missing_words_file, output_file="wordnet_mappings.json"):
        self.missing_words_file = missing_words_file
        self.output_file = output_file
        self.missing_words = []
        self.mappings = {}
        
        # Pre-defined dictionary of common splits for known compound words,
        # particularly useful for datasets like Pascal-Context 459.
        self.common_splits = {
            # Manually defined mappings
            "airconditioner": ["air", "conditioner"], "babycarriage": ["baby", "carriage"],
            "bambooweaving": ["bamboo", "weaving"], "baseballbat": ["baseball", "bat"],
            "basketballbackboard": ["basketball", "backboard"], "bottleopener": ["bottle", "opener"],
            "cabinetdoor": ["cabinet", "door"], "cameralens": ["camera", "lens"],
            "candleholder": ["candle", "holder"], "casetterecorder": ["casette", "recorder"],
            "cashregister": ["cash", "register"], "cdplayer": ["cd", "player"],
            "clothestree": ["clothes", "tree"], "coffeemachine": ["coffee", "machine"],
            "controlbooth": ["control", "booth"], "copyingmachine": ["copying", "machine"],
            "crabstick": ["crab", "stick"], "cuttingboard": ["cutting", "board"],
            "disccase": ["disc", "case"], "drainer": ["drain", "er"],
            "drinkdispenser": ["drink", "dispenser"], "drinkingmachine": ["drinking", "machine"],
            "drumkit": ["drum", "kit"], "electricfan": ["electric", "fan"],
            "electriciron": ["electric", "iron"], "electricpot": ["electric", "pot"],
            "electricsaw": ["electric", "saw"], "electronickeyboard": ["electronic", "keyboard"],
            "exhibitionbooth": ["exhibition", "booth"], "faxmachine": ["fax", "machine"],
            "ferriswheel": ["ferris", "wheel"], "fireextinguisher": ["fire", "extinguisher"],
            "firehydrant": ["fire", "hydrant"], "fishingnet": ["fishing", "net"],
            "fishingpole": ["fishing", "pole"], "fishtank": ["fish", "tank"],
            "gamecontroller": ["game", "controller"], "gamemachine": ["game", "machine"],
            "gascylinder": ["gas", "cylinder"], "gashood": ["gas", "hood"],
            "gasstove": ["gas", "stove"], "giftbox": ["gift", "box"],
            "glassmarble": ["glass", "marble"], "harddiskdrive": ["hard", "disk", "drive"],
            "horse-drawncarriage": ["horse-drawn", "carriage"], "hot-airballoon": ["hot-air", "balloon"],
            "hydrovalve": ["hydro", "valve"], "inflatorpump": ["inflator", "pump"],
            "ironingboard": ["ironing", "board"], "kart": ["kart"],  # Single words remain unchanged
            "kitchenrange": ["kitchen", "range"], "knifeblock": ["knife", "block"],
            "laddertruck": ["ladder", "truck"], "lifebuoy": ["life", "buoy"],
            "meterbox": ["meter", "box"], "musicalinstrument": ["musical", "instrument"],
            "oxygenbottle": ["oxygen", "bottle"], "paperbox": ["paper", "box"],
            "papercutter": ["paper", "cutter"], "pencontainer": ["pen", "container"],
            "pokerchip": ["poker", "chip"], "pooltable": ["pool", "table"],
            "pottedplant": ["potted", "plant"], "rangehood": ["range", "hood"],
            "recreationalmachines": ["recreational", "machines"], "remotecontrol": ["remote", "control"],
            "rockinghorse": ["rocking", "horse"], "sewingmachine": ["sewing", "machine"],
            "shoppingcart": ["shopping", "cart"], "signallight": ["signal", "light"],
            "speedbump": ["speed", "bump"], "spicecontainer": ["spice", "container"],
            "stickynote": ["sticky", "note"], "surveillancecamera": ["surveillance", "camera"],
            "swimmingpool": ["swimming", "pool"], "swimring": ["swim", "ring"],
            "telephonebooth": ["telephone", "booth"], "tong": ["tong"],  # Single words remain unchanged
            "toycar": ["toy", "car"], "trashbin": ["trash", "bin"],
            "tvmonitor": ["tv", "monitor"], "vacuumcleaner": ["vacuum", "cleaner"],
            "vendingmachine": ["vending", "machine"], "videocamera": ["video", "camera"],
            "videogameconsole": ["video", "game", "console"], "videoplayer": ["video", "player"],
            "washingmachine": ["washing", "machine"], "waterdispenser": ["water", "dispenser"],
            "waterpipe": ["water", "pipe"], "waterskateboard": ["water", "skateboard"],
            "windowblinds": ["window", "blinds"]
        }
    
    def load_missing_words(self):
        """Loads the list of OOV words from a text file."""
        with open(self.missing_words_file, 'r', encoding='utf-8') as f:
            self.missing_words = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(self.missing_words)} missing words.")
    
    def word_exists_in_wordnet(self, word):
        """Checks if a word exists in WordNet by attempting to find its synsets."""
        return len(wn.synsets(word)) > 0
    
    def create_mappings(self):
        """Iterates through missing words and attempts to generate a valid mapping for each."""
        logger.info("Starting to create mappings for OOV words...")
        
        for word in self.missing_words:
            # Attempt to find a valid split using a series of strategies
            mapped_parts = self.try_mapping_strategies(word)
            
            if mapped_parts:
                self.mappings[word] = mapped_parts
                logger.info(f"Created mapping for '{word}': {mapped_parts}")
            else:
                logger.warning(f"Failed to create a mapping for '{word}'")
        
        if self.missing_words:
            coverage = len(self.mappings) / len(self.missing_words) * 100
            logger.info(f"Successfully created {len(self.mappings)} mappings ({coverage:.1f}% coverage of missing words).")
    
    def try_mapping_strategies(self, word):
        """
        Applies a series of strategies to find a valid split for a word.
        The first strategy that succeeds is returned.
        """
        # Strategy 1: Check pre-defined mappings
        if word.lower() in self.common_splits:
            parts = self.common_splits[word.lower()]
            # Validate that each part exists in WordNet
            if all(self.word_exists_in_wordnet(part) for part in parts):
                return parts
            # If a part is not in WordNet, keep the mapping but log a warning
            logger.warning(f"Pre-defined mapping for '{word}' ({parts}) contains parts not in WordNet.")
            return parts
        
        # Strategy 2: Split by hyphen
        if '-' in word:
            parts = word.split('-')
            if all(self.word_exists_in_wordnet(part) for part in parts):
                return parts
        
        # Strategy 3: Split by common compound word patterns
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
                if all(self.word_exists_in_wordnet(part) for part in parts):
                    return parts
        
        # Strategy 4: Split by camelCase
        camel_case_parts = re.findall(r'[A-Z][a-z]*|[a-z]+', word)
        if len(camel_case_parts) > 1:
            lower_parts = [p.lower() for p in camel_case_parts]
            if all(self.word_exists_in_wordnet(part) for part in lower_parts):
                return lower_parts
        
        # Strategy 5: Recursive binary splitting (brute-force)
        for i in range(3, len(word) - 2):
            part1 = word[:i].lower()
            part2 = word[i:].lower()
            
            if self.word_exists_in_wordnet(part1) and self.word_exists_in_wordnet(part2):
                return [part1, part2]
                
        return None
    
    def save_mappings(self):
        """Saves the generated mappings to a JSON file."""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.mappings, f, indent=2, sort_keys=True)
        logger.info(f"Mappings have been saved to: {self.output_file}")
    
    def run(self):
        """Executes the entire mapping generation pipeline."""
        self.load_missing_words()
        self.create_mappings()
        self.save_mappings()
        self.analyze_results()
        return self.mappings
    
    def analyze_results(self):
        """Analyzes and reports on the results of the mapping process."""
        if not self.missing_words:
            logger.info("No missing words to analyze.")
            return

        total = len(self.missing_words)
        mapped = len(self.mappings)
        unmapped = total - mapped
        
        logger.info("Mapping Generation Analysis:")
        logger.info(f"- Total OOV words: {total}")
        logger.info(f"- Mapped: {mapped} ({mapped/total*100:.1f}%)")
        logger.info(f"- Unmapped: {unmapped} ({unmapped/total*100:.1f}%)")
        
        # Check how many mappings are fully valid (all parts in WordNet)
        valid_mappings = 0
        invalid_parts_log = []
        
        if mapped > 0:
            for word, parts in self.mappings.items():
                if all(self.word_exists_in_wordnet(part) for part in parts):
                    valid_mappings += 1
                else:
                    for part in parts:
                        if not self.word_exists_in_wordnet(part):
                            invalid_parts_log.append((word, part))
            
            logger.info(f"- Fully valid mappings: {valid_mappings} ({valid_mappings/mapped*100:.1f}% of all mappings)")
        
        if invalid_parts_log:
            logger.warning("The following mappings contain parts not found in WordNet:")
            for word, part in invalid_parts_log[:10]:  # Display first 10
                logger.warning(f"  - In '{word}': invalid part '{part}'")
            if len(invalid_parts_log) > 10:
                logger.warning(f"  ... and {len(invalid_parts_log) - 10} more.")
        
        # List the words that could not be mapped
        unmapped_words = [w for w in self.missing_words if w not in self.mappings]
        if unmapped_words:
            logger.warning("Could not find a mapping for the following words:")
            for word in unmapped_words[:10]:  # Display first 10
                logger.warning(f"  - {word}")
            if len(unmapped_words) > 10:
                logger.warning(f"  ... and {len(unmapped_words) - 10} more.")

if __name__ == "__main__":
    # Example usage: Generate mappings for missing WordNet words from the PC-459 dataset
    generator = WordNetMappingGenerator(
        missing_words_file="HS_OSM/word_missing_mapping/pc459_wordnet_missing.txt",
        output_file="HS_OSM/word_missing_mapping/wordnet_mappings.json"
    )
    mappings = generator.run()
