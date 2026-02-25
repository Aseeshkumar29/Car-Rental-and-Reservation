import re
import urllib.parse
from urllib.parse import urlparse
import numpy as np
from Levenshtein import distance as levenshtein_distance

class URLFeatureExtractor:
    def __init__(self):
        self.common_tlds = ['com', 'org', 'net', 'edu', 'gov', 'uk', 'de', 'fr']
        
    def extract_features(self, url):
        features = []
        
        # Parse URL
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        path = parsed_url.path.lower()
        query = parsed_url.query.lower()
        
        # 1. URL length
        features.append(len(url))
        
        # 2. Domain length
        features.append(len(domain))
        
        # 3. Number of subdomains
        subdomain_count = len([s for s in domain.split('.') if s])
        features.append(subdomain_count)
        
        # 4. Number of dots in domain
        features.append(domain.count('.'))
        
        # 5. Hyphens in domain
        features.append(domain.count('-'))
        
        # 6. Path length
        features.append(len(path))
        
        # 7. Query length
        features.append(len(query))
        
        # 8. Number of parameters in query
        query_params = len(parsed_url.query.split('&')) if parsed_url.query else 0
        features.append(query_params)
        
        # 9. Number of digits in URL
        features.append(sum(c.isdigit() for c in url))
        
        # 10. Number of special characters
        special_chars = set('!@#$%^&*()+=[]{}|;:,.<>?/~`')
        features.append(sum(c in special_chars for c in url))
        
        # 11. IP address in URL (binary)
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        features.append(1 if re.search(ip_pattern, url) else 0)
        
        # 12. URL shortening service
        shorteners = ['bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'buff.ly']
        features.append(1 if any(s in url.lower() for s in shorteners) else 0)
        
        # 13. Double slash in path
        features.append(1 if '//' in path else 0)
        
        # 14. Sensitive keywords
        sensitive_words = ['login', 'bank', 'paypal', 'account', 'password', 'security', 'verify']
        features.append(sum(1 for word in sensitive_words if word in url.lower()))
        
        # 15. Hexadecimal encoding
        features.append(1 if '%' in url and any(c.isdigit() for c in url.split('%')) else 0)
        
        # Pad features to fixed length
        while len(features) < 15:
            features.append(0)
            
        return np.array(features[:15], dtype=np.float32)