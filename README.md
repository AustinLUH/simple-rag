Instructions to install

curl -fsSL https://ollama.com/install.sh | sh  
pip install ollama  
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf  
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF  
python demo.py  


## Evaluation Results

| File used | Question no. | top_n | Generated answer | Correct? | If incorrect, how could it be fixed? |
|-----------|--------------|-------|------------------|----------|--------------------------------------|
| cat-facts.txt | 1 | 3 | Jacobson's organ (also known as vomeronasal organ). It is located in the upper surface of the mouth.| Yes |  |
| cat-facts.txt | 2 | 3 | Based on the information, a nine-year-old cat would have been awake for only 3 years out of its lifespan.  | Yes |  |
| cat-db-fiction-confusing.txt | 3 | 3 | Luna who must eat rabbit-based food is the Riverside Luna. She wears a yellow collar. | Yes |  |
| cat-db-fiction-confusing.txt | 4 | 3 | Based on the available information, we can conclude that Juniper cannot attend the adoption event scheduled for 2026-05-02 due to a veterinary note dated 2026-05-01 stating that Juniper must rest for seven days and cannot attend adoption events during that period. The evidence for this answer is primarily based on the veterinary note, which provides a clear restriction for Juniper. Since Jupiter has no activity restriction, it implies that Jupiter can also rest and attend the event safely. | Yes |  |
