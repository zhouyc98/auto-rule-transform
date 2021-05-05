import urllib
import pycurl
from io import BytesIO
import pandas as pd

query1 = '''PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX myclass: <http://www.semanticweb.org/16424/ontologies/2020/10/untitled-ontology-8#>
PREFIX : <http://www.semanticweb.org/16424/ontologies/2020/10/BuildingDesignFireCodesOntology#>
SELECT DISTINCT *
WHERE {
	?element1 rdf:type owl:NamedIndividual , myclass:Wall .
	?element1 :hasGlobalId ?element1_id .
	?element1 :hasFireResistanceLimits_hour ?dataproperty2 .
	BIND ((?dataproperty2 >= '3'^^xsd:decimal) AS ?Pass)
	FILTER (?Pass = 'false'^^xsd:boolean)
	}'''
query1 = urllib.parse.quote(query1)
query2 = urllib.parse.unquote(query1)

# url = "http://localhost:7200/repositories/FireCodeRepo?name=&infer=true&sameAs=false&query=select+*+where+%7B+%0A%09%3Fs+%3Fp+%3Fo+.%0A%7D+limit+100+%0A&execute="
url = "http://localhost:7200/repositories/FireCodeRepo?name=&infer=true&sameAs=false&query=" + query1
response_buffer = BytesIO()
curl = pycurl.Curl()
curl.setopt(curl.URL, url)
curl.setopt(curl.USERPWD, '%s:%s' % (' ', ' '))
curl.setopt(curl.WRITEFUNCTION, response_buffer.write)
curl.perform()
curl.close()
response_value = response_buffer.getvalue()
values = str(response_value, encoding="utf-8")
print(values)
data = values.split('\n')
postprocess_data = []
for line in data:
    if line:
        postprocess_data.append(line.rstrip().split(','))
df = pd.DataFrame(postprocess_data[1:], columns=postprocess_data[0])
print(df)

