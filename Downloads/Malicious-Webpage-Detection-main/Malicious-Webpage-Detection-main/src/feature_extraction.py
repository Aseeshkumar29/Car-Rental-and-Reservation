import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
import tldextract

# Load your existing data
df = pd.read_csv('data/urls_balanced_20k_per_class.csv')

# Basic cleaning
df = df.dropna(subset=['url', 'type'])
df['url'] = df['url'].str.strip()
df['type'] = df['type'].str.strip().str.lower()
df = df.drop_duplicates(subset=['url', 'type'])


ip_pattern = re.compile(r'^(?:\d{1,3}\.){3}\d{1,3}$')  # simple IPv4 regex

def parse_url_parts(url):
    try:
        parsed = urlparse(url)
        host = parsed.netloc.split(':')[0] 
        path = parsed.path
        return host, path
    except Exception:
        return "", ""

def is_ip(host):
    if not host:
        return False
    return bool(ip_pattern.match(host))

suspicious_keywords = [
    'login', 'verify', 'update', 'free', 'secure', 'bank',
    'account', 'password', 'confirm', 'signin', 'validate',
    'urgent', 'limited', 'win', 'prize'
]

def count_suspicious_words(url):
    lower = url.lower()
    return sum(1 for w in suspicious_keywords if w in lower)

# ---------- Create features ----------

# URL length
df['url_length'] = df['url'].str.len()

# Parse host and path
df['host'], df['path'] = zip(*df['url'].apply(parse_url_parts))

df['hostname_length'] = df['host'].str.len()
df['path_length'] = df['path'].str.len()

# Counts of characters
df['num_dots'] = df['url'].str.count(r'\.')
df['num_slashes'] = df['url'].str.count(r'/')
df['num_hyphens'] = df['url'].str.count(r'-')

df['num_digits'] = df['url'].apply(lambda u: sum(c.isdigit() for c in u))
df['num_letters'] = df['url'].apply(lambda u: sum(c.isalpha() for c in u))
df['num_special'] = df['url_length'] - df['num_digits'] - df['num_letters']
df['digit_ratio'] = df['num_digits'] / df['url_length']

# IP-related
df['has_ip'] = df['host'].apply(lambda h: 1 if is_ip(h) else 0)
df['ip_length'] = df['host'].apply(lambda h: len(h) if is_ip(h) else 0)

# Subdomain / domain / suffix
ext = df['url'].apply(lambda u: tldextract.extract(u))
df['subdomain'] = ext.apply(lambda e: e.subdomain)
df['domain'] = ext.apply(lambda e: e.domain)
df['suffix'] = ext.apply(lambda e: e.suffix)

def count_subdomains(sub):
    if not sub:
        return 0
    return sub.count('.') + 1  # 'a.b' -> 2
df['num_subdomains'] = df['subdomain'].apply(count_subdomains)

# Suspicious words
df['suspicious_word_count'] = df['url'].apply(count_suspicious_words)
df['has_suspicious_word'] = (df['suspicious_word_count'] > 0).astype(int)

# ---------- Save extended CSV ----------

df.to_csv('data/urls_with_features.csv', index=False)
print("Saved extended dataset to data/interim/urls_with_features.csv")