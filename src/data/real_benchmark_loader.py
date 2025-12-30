"""
PRODUCTION REAL DATA LOADER
Loads ONLY real benchmark datasets from HuggingFace
NO MOCKS, NO FALLBACKS, NO SYNTHETIC

Requires: HF_TOKEN environment variable
"""

import os
import logging
from typing import Dict, List, Optional
import pandas as pd
from datasets import load_dataset
from huggingface_hub import login

logger = logging.getLogger(__name__)


class ProductionDataLoader:
    """
    Production data loader - REAL data only
    
    Datasets loaded from HuggingFace:
    - JailbreakBench (jailbreak detection)
    - WildGuardMix (AllenAI safety)
    - ToxicChat (benign samples)
    - Additional real attack datasets
    
    NO SYNTHETIC FALLBACKS
    """
    
    def __init__(self, hf_token: Optional[str] = None):
        """
        Initialize production data loader
        
        Args:
            hf_token: HuggingFace token (or reads from HF_TOKEN env var)
        """
        # Get HF token
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        
        if not self.hf_token:
            raise ValueError(
                "HF_TOKEN not found! Set environment variable: export HF_TOKEN='your_token_here'"
            )
        
        # Login to HuggingFace
        try:
            login(token=self.hf_token, add_to_git_credential=False)
            logger.info("✅ Logged into HuggingFace")
        except Exception as e:
            raise RuntimeError(f"Failed to login to HuggingFace: {e}")
        
        # Label mapping (standardized)
        self.label_map = {
            'benign': 0,
            'safe': 0,
            'direct_injection': 1,
            'prompt_injection': 1,
            'injection': 1,
            'jailbreak': 2,
            'obfuscation': 3,
            'unsafe': 1,  # Map unsafe to attack
        }
    
    def load_jailbreakbench(self) -> pd.DataFrame:
        """
        Load JailbreakBench from HuggingFace
        
        Dataset: JailbreakBench/JBB-Behaviors
        - 100 harmful behaviors + 100 benign
        - Industry standard jailbreak benchmark
        
        Returns:
            DataFrame with real jailbreak data
        """
        logger.info("📥 Loading JailbreakBench from HuggingFace...")
        
        try:
            # Load both harmful and benign splits
            harmful_dataset = load_dataset(
                "JailbreakBench/JBB-Behaviors",
                "behaviors",
                split="harmful",
                token=self.hf_token,
            )
            
            benign_dataset = load_dataset(
                "JailbreakBench/JBB-Behaviors",
                "behaviors",
                split="benign",
                token=self.hf_token,
            )
            
            data = []
            
            # Process harmful behaviors (jailbreak attempts)
            for sample in harmful_dataset:
                # Try different possible field names
                behavior = sample.get('Behavior') or sample.get('behavior') or sample.get('text', '')
                category = sample.get('Category') or sample.get('category', 'unknown')
                
                data.append({
                    'text': behavior,
                    'label': 2,  # jailbreak
                    'attack_type': 'jailbreak',
                    'source': 'jailbreakbench',
                    'category': category,
                })
            
            # Process benign behaviors
            for sample in benign_dataset:
                # Try different possible field names
                behavior = sample.get('Behavior') or sample.get('behavior') or sample.get('text', '')
                category = sample.get('Category') or sample.get('category', 'unknown')
                
                data.append({
                    'text': behavior,
                    'label': 0,  # benign
                    'attack_type': 'benign',
                    'source': 'jailbreakbench',
                    'category': category,
                })
            
            df = pd.DataFrame(data)
            logger.info(f"✅ Loaded {len(df):,} JailbreakBench samples ({len(harmful_dataset)} harmful + {len(benign_dataset)} benign)")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to load JailbreakBench: {e}")
            raise RuntimeError(f"Cannot load JailbreakBench. Check HF_TOKEN and dataset access: {e}")
    
    def load_wildguard(self, max_samples: int = 20000) -> pd.DataFrame:
        """
        Load WildGuardMix from AllenAI
        
        Dataset: allenai/wildguardmix
        - Safety classification dataset
        - Use 'safe' samples as benign, 'unsafe' as attacks
        
        Returns:
            DataFrame with WildGuard data
        """
        logger.info("📥 Loading WildGuardMix from HuggingFace...")
        
        try:
            # Load from HuggingFace
            dataset = load_dataset(
                "allenai/wildguardmix",
                "wildguardtrain",  # Config name for train split
                split="train",
                token=self.hf_token,
            )
            
            data = []
            for sample in dataset:
                if len(data) >= max_samples:
                    break
                
                # Extract prompt and label
                prompt = sample.get('prompt', '')
                label_str = sample.get('label', 'safe').lower()
                
                # Map to our labels
                if 'safe' in label_str or 'benign' in label_str:
                    label = 0  # benign
                elif 'jailbreak' in label_str:
                    label = 2  # jailbreak
                else:
                    label = 1  # generic attack
                
                data.append({
                    'text': prompt,
                    'label': label,
                    'attack_type': 'benign' if label == 0 else 'attack',
                    'source': 'wildguard',
                })
            
            df = pd.DataFrame(data)
            logger.info(f"✅ Loaded {len(df):,} WildGuardMix samples")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to load WildGuardMix: {e}")
            raise RuntimeError(f"Cannot load WildGuardMix. Check HF_TOKEN and dataset access: {e}")
    
    def load_toxicchat(self, max_samples: int = 10000) -> pd.DataFrame:
        """
        Load ToxicChat benign samples
        
        Dataset: lmsys/toxic-chat
        - Real user-chatbot conversations
        - Filter for benign (non-toxic, non-jailbreaking) samples
        
        Returns:
            DataFrame with benign samples
        """
        logger.info("📥 Loading ToxicChat from HuggingFace...")
        
        try:
            # Load from HuggingFace
            dataset = load_dataset(
                "lmsys/toxic-chat",
                "toxicchat0124",
                split="train",
                token=self.hf_token,
            )
            
            # Filter benign samples
            data = []
            for sample in dataset:
                if len(data) >= max_samples:
                    break
                
                # Only keep benign samples (toxicity=0, jailbreaking=0)
                if sample.get('toxicity', 1) == 0 and sample.get('jailbreaking', 1) == 0:
                    data.append({
                        'text': sample['user_input'],
                        'label': 0,  # benign
                        'attack_type': 'benign',
                        'source': 'toxicchat',
                    })
            
            df = pd.DataFrame(data)
            logger.info(f"✅ Loaded {len(df):,} ToxicChat benign samples")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to load ToxicChat: {e}")
            raise RuntimeError(f"Cannot load ToxicChat. Check HF_TOKEN and dataset access: {e}")
    
    def load_prompt_injections(self, max_samples: int = 5000) -> pd.DataFrame:
        """
        Load real prompt injection datasets
        
        Dataset: deepset/prompt-injections
        - Real prompt injection examples
        - Industry benchmark
        
        Returns:
            DataFrame with prompt injection data
        """
        logger.info("📥 Loading Prompt Injections from HuggingFace...")
        
        try:
            # Load from HuggingFace
            dataset = load_dataset(
                "deepset/prompt-injections",
                split="train",
                token=self.hf_token,
            )
            
            data = []
            for sample in dataset:
                if len(data) >= max_samples:
                    break
                
                # Handle both string and integer labels
                label_raw = sample.get('label', 1)
                
                if isinstance(label_raw, (int, float)):
                    # Integer label: 0 = benign, 1 (or >0) = injection
                    label = 1 if label_raw > 0 else 0
                elif isinstance(label_raw, str):
                    # String label: parse text
                    label_str = label_raw.lower()
                    if 'injection' in label_str or 'attack' in label_str:
                        label = 1  # direct_injection
                    elif 'benign' in label_str or 'safe' in label_str:
                        label = 0  # benign
                    else:
                        label = 1  # default to attack
                else:
                    # Unknown type, default to attack
                    label = 1
                
                data.append({
                    'text': sample.get('text', sample.get('prompt', '')),
                    'label': label,
                    'attack_type': 'direct_injection' if label == 1 else 'benign',
                    'source': 'prompt_injections',
                })
            
            df = pd.DataFrame(data)
            logger.info(f"✅ Loaded {len(df):,} Prompt Injection samples")
            
            return df
            
        except Exception as e:
            logger.warning(f"⚠️  Could not load prompt injections dataset: {e}")
            # Return empty DataFrame instead of raising
            return pd.DataFrame(columns=['text', 'label', 'attack_type', 'source'])
    
    def load_all_real_data(self) -> pd.DataFrame:
        """
        Load ALL available real benchmark data
        
        NO SYNTHETIC, NO MOCKS, NO FALLBACKS
        
        Returns:
            Combined DataFrame with all real data
        """
        logger.info("\n" + "="*80)
        logger.info("📊 LOADING PRODUCTION DATA (REAL BENCHMARKS ONLY)")
        logger.info("="*80)
        
        all_data = []
        
        # 1. JailbreakBench (REQUIRED)
        jbb_data = self.load_jailbreakbench()
        all_data.append(jbb_data)
        
        # 2. WildGuard (REQUIRED)
        wildguard_data = self.load_wildguard(max_samples=20000)
        all_data.append(wildguard_data)
        
        # 3. ToxicChat (REQUIRED)
        toxicchat_data = self.load_toxicchat(max_samples=10000)
        all_data.append(toxicchat_data)
        
        # 4. Prompt Injections (OPTIONAL - may not have access)
        try:
            pi_data = self.load_prompt_injections(max_samples=5000)
            if len(pi_data) > 0:
                all_data.append(pi_data)
        except:
            logger.warning("⚠️  Skipping prompt injections dataset (not accessible)")
        
        # Combine all data
        combined = pd.concat(all_data, ignore_index=True)
        
        # Shuffle
        combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("📊 REAL DATA SUMMARY")
        logger.info("="*80)
        logger.info(f"Total samples:     {len(combined):,}")
        logger.info(f"\nLabel distribution:")
        for label, count in combined['label'].value_counts().sort_index().items():
            label_names = ['Benign', 'Direct Injection', 'Jailbreak', 'Obfuscation']
            pct = count / len(combined) * 100
            logger.info(f"  {label_names[label]:20s}: {count:,} ({pct:.1f}%)")
        
        logger.info(f"\nSource distribution:")
        for source, count in combined['source'].value_counts().items():
            pct = count / len(combined) * 100
            logger.info(f"  {source:20s}: {count:,} ({pct:.1f}%)")
        logger.info("="*80 + "\n")
        
        if len(combined) < 10000:
            raise RuntimeError(
                f"Insufficient data loaded ({len(combined)} samples). "
                "Check HF_TOKEN and dataset access permissions."
            )
        
        return combined


def verify_hf_access():
    """
    Verify HuggingFace access and token
    
    Raises:
        RuntimeError if token is invalid or datasets are inaccessible
    """
    hf_token = os.getenv('HF_TOKEN')
    
    if not hf_token:
        raise RuntimeError(
            "\n❌ HF_TOKEN not found!\n\n"
            "Set your HuggingFace token:\n"
            "  export HF_TOKEN='hf_your_token_here'\n\n"
            "Get your token at: https://huggingface.co/settings/tokens\n"
        )
    
    logger.info("✅ HF_TOKEN found in environment")
    
    # Test login
    try:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        logger.info("✅ HuggingFace login successful")
    except Exception as e:
        raise RuntimeError(f"❌ HuggingFace login failed: {e}")
    
    # Test dataset access
    try:
        test_ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train[:10]", token=hf_token)
        logger.info(f"✅ Dataset access verified (loaded {len(test_ds)} test samples)")
    except Exception as e:
        raise RuntimeError(f"❌ Dataset access failed: {e}")
    
    logger.info("\n🎉 HuggingFace access verified successfully!\n")


if __name__ == "__main__":
    # Verify access
    verify_hf_access()
    
    # Load all real data
    loader = ProductionDataLoader()
    data = loader.load_all_real_data()
    
    print(f"\n✅ Loaded {len(data):,} real samples from HuggingFace")
    print(data.head())

