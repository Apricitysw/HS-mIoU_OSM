import networkx as nx
import numpy as np
from nltk.corpus import wordnet as wn
import logging
import os
from scipy.spatial.distance import cosine
import pandas as pd
import json
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HierarchicalSimilarity:
    """
    Calculates a hierarchy-aware similarity between words by combining GloVe embeddings 
    for semantic similarity and the WordNet hierarchy for structural relationships.

    Args:
        gamma (float): Reward factor, controlling the magnitude of the reward for hierarchical proximity.
        beta (float): Penalty factor, controlling the magnitude of the penalty for hierarchical distance.
        delta (int): Threshold distance. Distances below this value are rewarded, while those above are penalized.
        alpha (float): Modifier influence weight, controlling the impact of modifier similarity in compound words.
        glove_path (str): Path to the pre-trained GloVe word vector file.
        dimensions (int): Dimension of the GloVe word vectors.
        glove_mapping_file (str, optional): Path to a JSON file for mapping OOV (Out-of-Vocabulary) words for GloVe.
        wordnet_mapping_file (str, optional): Path to a JSON file for mapping OOV words for WordNet.
        use_which_adj (int): Selects the adjustment formula to use (0 or 1).
    """
    def __init__(self, gamma=0.3, beta=0.5, delta=4, alpha=0.4, glove_path='glove.6B.300d.txt', 
                    dimensions=300, glove_mapping_file=None, wordnet_mapping_file=None, use_which_adj=0):
        self.gamma = gamma
        self.beta = beta
        self.delta = delta
        self.alpha = alpha
        self.word_vectors = {}
        self.dimensions = dimensions
        self.use_which_adj = use_which_adj
        
        # Tracking sets for Out-of-Vocabulary (OOV) words
        self.glove_missing_words = set()
        self.wordnet_missing_words = set()
        
        # Load word mapping files for handling OOV words
        self.glove_mappings = self._load_mapping_file(glove_mapping_file, "GloVe")
        self.wordnet_mappings = self._load_mapping_file(wordnet_mapping_file, "WordNet")

        # Pre-calculate and cache the maximum depth of the WordNet hierarchy for normalization
        self.max_depth = max(len(path) for syn in wn.all_synsets() for path in syn.hypernym_paths())
        logger.info(f"Maximum WordNet depth: {self.max_depth}")

        # Load GloVe word vectors
        self._load_glove(glove_path)

    def _load_mapping_file(self, file_path, name):
        """Loads a JSON mapping file."""
        mappings = {}
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    mappings = json.load(f)
                logger.info(f"Loaded {len(mappings)} {name} word mappings from {file_path}.")
            except Exception as e:
                logger.error(f"Failed to load {name} word mappings: {str(e)}")
        return mappings

    def _load_glove(self, glove_path):
        """Loads pre-trained GloVe word vectors using pandas for efficiency."""
        logger.info(f"Loading GloVe word vectors from: {glove_path}")
        try:
            # Determine file size to decide whether to use chunked reading for large files
            file_size_gb = os.path.getsize(glove_path) / 1e9
            
            if file_size_gb > 2:  # Use chunked reading for files larger than 2GB
                logger.info(f"File size is {file_size_gb:.1f}GB, using chunked reading.")
                self.word_vectors = {}
                chunk_size = 100000  # Number of lines to read per chunk
                
                # Read a small portion to determine vector dimensions
                temp_df = pd.read_csv(glove_path, sep=' ', nrows=1000, header=None, quoting=3, on_bad_lines='skip')
                vector_dim = temp_df.shape[1] - 1
                
                for chunk in pd.read_csv(glove_path, sep=' ', chunksize=chunk_size, 
                                        header=None, quoting=3, on_bad_lines='skip'):
                    for _, row in chunk.iterrows():
                        try:
                            word = row[0]
                            vector = row[1:vector_dim+1].values.astype('float32')
                            if len(vector) == self.dimensions:
                                self.word_vectors[word] = vector
                        except (ValueError, IndexError):
                            continue
            else: # Read smaller files directly
                df = pd.read_csv(glove_path, sep=' ', header=None, quoting=3, on_bad_lines='skip')
                vector_dim = df.shape[1] - 1
                
                for _, row in df.iterrows():
                    try:
                        word = row[0]
                        vector = row[1:vector_dim+1].values.astype('float32')
                        if len(vector) == self.dimensions:
                            self.word_vectors[word] = vector
                    except (ValueError, IndexError):
                        continue
                        
            logger.info(f"GloVe vectors loaded. Total words: {len(self.word_vectors)}")
        except Exception as e:
            logger.error(f"Failed to load GloVe vectors: {str(e)}")
            raise

    def get_text_similarity(self, text1, text2):
        """
        Calculates the cosine similarity between two text strings using their GloVe vectors.
        Returns 0.0 if either text cannot be vectorized.
        """
        try:
            vec1 = self._get_word_vector(text1)
            vec2 = self._get_word_vector(text2)
            
            if vec1 is None or vec2 is None:
                return 0.0
            
            # Cosine similarity is 1 - cosine distance
            similarity = 1 - cosine(vec1, vec2)
            return similarity
        except Exception as e:
            logger.error(f"Error calculating text similarity for '{text1}' vs '{text2}': {str(e)}")
            return 0.0

    def _is_compound_word(self, word):
        """Determines if a word is a compound word based on separators or mappings."""
        if ' ' in word or '-' in word:
            return True
        
        # Check if the word is defined as a multi-part mapping
        if word in self.wordnet_mappings and len(self.wordnet_mappings[word]) > 1:
            return True
        if word in self.glove_mappings and len(self.glove_mappings[word]) > 1:
            return True
        
        # Check for camelCase
        if len(re.findall(r'[A-Z][a-z]*', word)) > 1:
            return True
            
        return False

    def _get_head_word(self, word):
        """Extracts the head noun from a (potentially compound) word."""
        if ' ' in word:
            return word.split()[-1]
        if '-' in word:
            return word.split('-')[-1]
        
        if word in self.wordnet_mappings:
            return self.wordnet_mappings[word][-1]
        if word in self.glove_mappings:
            return self.glove_mappings[word][-1]
        
        # Handle camelCase
        parts = re.findall(r'[A-Z][a-z]*|[a-z]+', word)
        if len(parts) > 1:
            return parts[-1].lower()
        
        return word

    def _get_modifier_words(self, word):
        """Extracts the modifier(s) from a compound word."""
        if ' ' in word:
            parts = word.split()
            return " ".join(parts[:-1]) if len(parts) > 1 else None
        if '-' in word:
            parts = word.split('-')
            return "-".join(parts[:-1]) if len(parts) > 1 else None

        if word in self.wordnet_mappings:
            parts = self.wordnet_mappings[word]
            return " ".join(parts[:-1]) if len(parts) > 1 else None
        if word in self.glove_mappings:
            parts = self.glove_mappings[word]
            return " ".join(parts[:-1]) if len(parts) > 1 else None
        
        # Handle camelCase
        parts = re.findall(r'[A-Z][a-z]*|[a-z]+', word)
        if len(parts) > 1:
            return " ".join(parts[:-1]).lower()
        
        return None

    def _get_word_vector(self, word):
        """
        Retrieves the GloVe vector for a word, with fallbacks for case, phrases, and mappings.
        """
        # Direct hit
        if word in self.word_vectors:
            return self.word_vectors[word]
        
        # Try lowercase
        word_lower = word.lower()
        if word_lower in self.word_vectors:
            return self.word_vectors[word_lower]
            
        # Handle multi-word phrases by averaging vectors
        if ' ' in word:
            parts = word.split()
            vectors = [self._get_word_vector(part) for part in parts]
            valid_vectors = [v for v in vectors if v is not None]
            if valid_vectors:
                return np.mean(valid_vectors, axis=0)
        
        # Try to resolve using pre-defined mappings
        mapping_key = word if word in self.glove_mappings else word_lower
        if mapping_key in self.glove_mappings:
            parts = self.glove_mappings[mapping_key]
            vectors = [self._get_word_vector(part) for part in parts]
            valid_vectors = [v for v in vectors if v is not None]
            if valid_vectors:
                logger.info(f"Constructed vector for '{word}' using mapping: {parts}")
                return np.mean(valid_vectors, axis=0)
        
        # If all fallbacks fail, log as missing and return None
        self.glove_missing_words.add(word)
        logger.warning(f"Word not found in GloVe vocabulary: '{word}'")
        return None

    def _get_synset(self, word):
        """
        Retrieves the most likely WordNet synset for a word, with fallbacks.
        """
        # Try direct lookup
        syns = wn.synsets(word)
        if syns:
            return syns[0]

        # Try lowercase
        word_lower = word.lower()
        syns = wn.synsets(word_lower)
        if syns:
            return syns[0]

        # For phrases, search for parts in reverse order
        if ' ' in word:
            for part in reversed(word.split()):
                syn = self._get_synset(part)
                if syn:
                    return syn
                
        # If not found, log as missing
        self.wordnet_missing_words.add(word)
        return None

    def _calculate_adjusted_depth(self, word, synset):
        """Calculates hierarchical depth, adding 1 for compound words."""
        if synset is None:
            return 0
        
        base_depth = len(synset.hypernym_paths()[0])
        
        # Compound words are considered one level deeper than their head word
        return base_depth + 1 if self._is_compound_word(word) else base_depth

    def _calculate_adjusted_distance(self, word1, word2, syn1, syn2):
        """Calculates WordNet path distance, adjusted for compound words."""
        if syn1 is None or syn2 is None:
            return None
        
        base_distance = syn1.shortest_path_distance(syn2)
        if base_distance is None:
            return None
        
        is_compound1 = self._is_compound_word(word1)
        is_compound2 = self._is_compound_word(word2)
        
        # Adjust distance based on whether words are compound
        if is_compound1 and is_compound2:
            return base_distance + 2
        elif is_compound1 or is_compound2:
            return base_distance + 1
        else:
            return base_distance

    def get_sim(self, word1, word2):
        """
        Calculates the final hierarchy-aware similarity between two words.
        This method dispatches to different logic based on whether the words are
        simple, compound, or a mix.

        Returns:
            tuple[float, float]: A tuple containing the final adjusted similarity and the base semantic similarity.
        """
        if word1.lower() == word2.lower():
            return 1.0, 1.0
        
        is_compound1 = self._is_compound_word(word1)
        is_compound2 = self._is_compound_word(word2)
        
        try:
            # Case 1: Two simple words
            if not is_compound1 and not is_compound2:
                s_base = self.get_text_similarity(word1, word2)
                syn1 = self._get_synset(word1)
                syn2 = self._get_synset(word2)
            
            # Case 2: Two compound words
            elif is_compound1 and is_compound2:
                head1, head2 = self._get_head_word(word1), self._get_head_word(word2)
                modifier1, modifier2 = self._get_modifier_words(word1), self._get_modifier_words(word2)
                
                s_head = self.get_text_similarity(head1, head2)
                
                if modifier1 and modifier2:
                    s_modifier = self.get_text_similarity(modifier1, modifier2)
                elif modifier1 or modifier2:
                    s_modifier = 0.1 # Penalize if one has a modifier and the other doesn't
                else:
                    s_modifier = 1.0 # Should not happen, but for safety
                
                s_base = (1 - self.alpha) * s_head + self.alpha * s_modifier
                syn1, syn2 = self._get_synset(head1), self._get_synset(head2)

            # Case 3: One simple and one compound word
            else:
                if is_compound1:
                    compound_word, single_word = word1, word2
                else:
                    compound_word, single_word = word2, word1
                
                head = self._get_head_word(compound_word)
                modifier = self._get_modifier_words(compound_word)
                
                s_head = self.get_text_similarity(single_word, head)
                
                s_modifier_contribution = 0.0
                if modifier:
                    s_mod_to_single = self.get_text_similarity(modifier, single_word)
                    mod_vec = self._get_word_vector(modifier)
                    # Modifier strength can be approximated by its vector norm
                    mod_strength = min(np.linalg.norm(mod_vec) / 10.0, 1.0) if mod_vec is not None else 0.5
                    s_modifier_contribution = s_mod_to_single * mod_strength * self.alpha
                
                s_base = min(s_head + s_modifier_contribution, 1.0)
                syn1, syn2 = self._get_synset(single_word), self._get_synset(head)

            # Apply hierarchical adjustment if synsets were found
            if syn1 and syn2:
                L = self._calculate_adjusted_distance(word1, word2, syn1, syn2)
                if L is not None:
                    D1 = self._calculate_adjusted_depth(word1, syn1)
                    D2 = self._calculate_adjusted_depth(word2, syn2)
                    D_avg = (D1 + D2) / 2
                    D_rel = abs(D1 - D2) + 1e-5 # Add epsilon to avoid division by zero
                    
                    adjustment = [0, 0]
                    # Formula 1
                    factor = (self.gamma + self.beta) / 2
                    adjustment[0] = np.tanh(factor * (self.delta - L) * (D_avg / (D_rel * self.max_depth)))
                    # Formula 2
                    adjustment[1] = np.exp(self.gamma * (D_avg - L) / D_avg / (1 + self.beta * max(0, (L - self.delta)) * D_rel)) - 1
                    
                    final_sim = s_base * (1 + adjustment[self.use_which_adj])
                    final_sim = max(0.0, min(1.0, final_sim)) # Clip to [0, 1]
                    return final_sim, s_base
            
            # Fallback to base similarity if no hierarchical info is available
            return s_base, s_base

        except Exception as e:
            logger.error(f"Error calculating similarity for '{word1}' vs '{word2}': {str(e)}")
            # Fallback to basic text similarity on error
            return self.get_text_similarity(word1, word2), self.get_text_similarity(word1, word2)

    def batch_compute_similarity(self, words_list):
        """Computes a similarity matrix for all pairs in a list of words."""
        n = len(words_list)
        sim_matrix = np.zeros((n, n))
        
        for i in range(n):
            sim_matrix[i, i] = 1.0
            for j in range(i + 1, n):
                sim, _ = self.get_sim(words_list[i], words_list[j])
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim # Matrix is symmetric
                
        return sim_matrix

    def save_missing_words(self, glove_file="glove_missing_words.txt", wordnet_file="wordnet_missing_words.txt"):
        """Saves the sets of missing words to text files."""
        if self.glove_missing_words:
            with open(glove_file, 'w') as f:
                for word in sorted(list(self.glove_missing_words)):
                    f.write(word + '\n')
            logger.info(f"Saved {len(self.glove_missing_words)} missing GloVe words to {glove_file}")

        if self.wordnet_missing_words:
            with open(wordnet_file, 'w') as f:
                for word in sorted(list(self.wordnet_missing_words)):
                    f.write(word + '\n')
            logger.info(f"Saved {len(self.wordnet_missing_words)} missing WordNet words to {wordnet_file}")

    def save_similarity_matrix(self, words_list, output_file):
        """Computes and saves the similarity matrix to a CSV file."""
        sim_matrix = self.batch_compute_similarity(words_list)
        df = pd.DataFrame(sim_matrix, index=words_list, columns=words_list)
        df.to_csv(output_file)
        logger.info(f"Similarity matrix saved to: {output_file}")

