import json
from service import diagnose
import sys
sys.stdout.reconfigure(encoding='utf-8')


symptoms = "fever, cough, sore throat, body pain"

result = diagnose(symptoms)

print(json.dumps(result, indent=2, ensure_ascii=False))

