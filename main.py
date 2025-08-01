import os
import json
import xml.etree.ElementTree as ET
from flask import Flask, request, render_template_string
from google.cloud import aiplatform

# Initialize the Flask application
app = Flask(__name__)

# Initialize Vertex AI client
# The VM's service account credentials will be used for authentication.
# Update 'your-gcp-project-id' with your actual project ID.
PROJECT_ID = "your-gcp-project-id"
REGION = "us-central1" # Or another region where Vertex AI is available

# The model name for Gemini 2.0
MODEL_NAME = "gemini-2.0"

# Initialize the Vertex AI client
aiplatform.init(project=PROJECT_ID, location=REGION)

def generate_bigquery_sql_from_xml(xml_string):
    """
    Parses an XML string and uses a single Vertex AI Gemini 2.0 call
    to generate the BigQuery SQL query. This is the 'without A2A' method.

    Args:
        xml_string (str): The XML input containing the query details.

    Returns:
        tuple: (Generated SQL string, None) or (Error message, None).
    """
    try:
        root = ET.fromstring(xml_string)
        
        # Extract information from the XML
        action = root.find("query").find("action").text if root.find("query").find("action") is not None else "SELECT"
        table = root.find("query").find("table").text if root.find("query").find("table") is not None else ""
        columns_element = root.find("query").find("columns")
        columns = [c.text for c in columns_element.findall('column')] if columns_element is not None else ["*"]
        columns_str = ", ".join(columns) if columns else "*"

        conditions = []
        conditions_element = root.find("query").find("conditions")
        if conditions_element is not None:
            for condition in conditions_element.findall('condition'):
                field = condition.find('field').text
                operator = condition.find('operator').text
                value = condition.find('value').text
                conditions.append(f"{field} {operator} '{value}'")
        conditions_str = " AND ".join(conditions) if conditions else ""

        # Construct a direct prompt for the language model
        prompt = f"""
        You are a helpful agent that converts structured XML prompts into a valid BigQuery SQL query.
        You must only respond with the BigQuery SQL code and nothing else.

        XML Prompt Details:
        - Action: {action}
        - Table: `{table}`
        - Columns: {columns_str}
        - Conditions: {conditions_str}

        Please generate a single BigQuery SQL query based on these details.
        
        Example:
        If the XML details are:
        - Action: SELECT
        - Table: `mydataset.employees`
        - Columns: first_name, last_name
        - Conditions: department_id = '101'
        The BigQuery SQL query should be: SELECT first_name, last_name FROM `mydataset.employees` WHERE department_id = '101';
        """

        model = aiplatform.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        
        sql_query = response.text.strip().replace("```sql\n", "").replace("```", "")
        return sql_query, None

    except ET.ParseError as e:
        return None, f"Error parsing XML: {e}"
    except Exception as e:
        return None, f"An error occurred: {e}"

def generate_plan_from_xml(xml_string):
    """
    Uses the Vertex AI Gemini 2.0 model as a 'Planner Agent' to convert
    XML into a structured JSON plan.
    """
    prompt = f"""
    You are a planning agent. Your task is to extract the details from an XML string and
    return a structured JSON object. The JSON object should contain the 'table', 'columns',
    and 'conditions' from the XML. Do not respond with anything other than the JSON object.

    XML Input:
    {xml_string}
    
    Example output JSON:
    ```json
    {{
      "table": "mydataset.orders",
      "columns": ["order_id", "customer_name"],
      "conditions": [
        {{"field": "customer_id", "operator": "=", "value": "123"}},
        {{"field": "status", "operator": "=", "value": "shipped"}}
      ]
    }}
    ```
    """
    try:
        model = aiplatform.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        json_plan = response.text.strip().replace("```json\n", "").replace("```", "")
        return json.loads(json_plan), None
    except Exception as e:
        return None, f"Error generating plan: {e}"

def generate_sql_from_plan(plan):
    """
    Uses the Vertex AI Gemini 2.0 model as a 'Generator Agent' to create
    a BigQuery SQL query from a structured JSON plan.
    """
    try:
        table = plan.get('table', '')
        columns = plan.get('columns', [])
        columns_str = ", ".join(columns) if columns else "*"

        conditions = plan.get('conditions', [])
        conditions_str = ""
        if conditions:
            formatted_conditions = []
            for cond in conditions:
                field = cond.get('field')
                operator = cond.get('operator')
                value = cond.get('value')
                formatted_conditions.append(f"{field} {operator} '{value}'")
            conditions_str = " AND ".join(formatted_conditions)

        prompt = f"""
        You are a SQL generator agent. Your task is to take a structured JSON plan
        and generate a valid BigQuery SQL query. Do not respond with anything other than the SQL code.

        JSON Plan:
        {json.dumps(plan, indent=2)}

        Generate the BigQuery SQL query based on this plan.
        """
        
        model = aiplatform.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        sql_query = response.text.strip().replace("```sql\n", "").replace("```", "")
        return sql_query, None
    except Exception as e:
        return None, f"Error generating SQL from plan: {e}"


# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>XML to BigQuery SQL Agent</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { font-family: 'Inter', sans-serif; }
    </style>
</head>
<body class="bg-gray-100 flex items-center justify-center min-h-screen p-4">
    <div class="bg-white p-8 rounded-xl shadow-lg w-full max-w-4xl">
        <h1 class="text-3xl font-bold text-center text-gray-800 mb-6">XML to BigQuery SQL Agent</h1>
        <p class="text-center text-gray-600 mb-8">
            Paste your XML prompt below to generate a BigQuery SQL query.
        </p>

        <form action="/generate_sql" method="post" class="space-y-6">
            <div>
                <label for="xml_input" class="block text-sm font-medium text-gray-700">XML Input</label>
                <textarea id="xml_input" name="xml_input" rows="10" 
                          class="mt-1 block w-full rounded-md border-gray-300 shadow-sm 
                          focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-3 border-2"
                          placeholder="<request><query><action>SELECT</action><table>mydataset.orders</table><columns>...</columns></query></request>"></textarea>
            </div>
            <div class="flex items-center justify-between">
                <div class="flex items-center">
                    <input id="use_a2a" name="use_a2a" type="checkbox"
                           class="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded">
                    <label for="use_a2a" class="ml-2 block text-sm text-gray-900">Use Agent-to-Agent Protocol</label>
                </div>
                <button type="submit" 
                        class="flex justify-center py-2 px-4 border border-transparent 
                               rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 
                               hover:bg-indigo-700 focus:outline-none focus:ring-2 
                               focus:ring-offset-2 focus:ring-indigo-500">
                    Generate SQL
                </button>
            </div>
        </form>

        {% if error_message %}
        <div class="mt-8 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
            <span class="block sm:inline">{{ error_message }}</span>
        </div>
        {% endif %}

        {% if sql_query_a2a %}
        <div class="mt-8">
            <h2 class="text-xl font-semibold text-gray-800 mb-3">Output with Agent-to-Agent Protocol</h2>
            <div class="mb-6">
                <h3 class="text-lg font-medium text-gray-700 mb-2">Generated Plan (Intermediate Step)</h3>
                <div class="bg-gray-800 text-gray-200 rounded-md p-4 overflow-x-auto shadow-inner">
                    <pre><code class="whitespace-pre-wrap">{{ json_plan | tojson(indent=2) }}</code></pre>
                </div>
            </div>
            <div>
                <h3 class="text-lg font-medium text-gray-700 mb-2">Generated BigQuery SQL</h3>
                <div class="bg-gray-800 text-gray-200 rounded-md p-4 overflow-x-auto shadow-inner">
                    <pre><code class="whitespace-pre-wrap">{{ sql_query_a2a }}</code></pre>
                </div>
            </div>
        </div>
        {% endif %}
        
        {% if sql_query_direct %}
        <div class="mt-8">
            <h2 class="text-xl font-semibold text-gray-800 mb-3">Output without Agent-to-Agent Protocol</h2>
            <div class="bg-gray-800 text-gray-200 rounded-md p-4 overflow-x-auto shadow-inner">
                <pre><code class="whitespace-pre-wrap">{{ sql_query_direct }}</code></pre>
            </div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

# Route for the main page
@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE)

# Route to handle the form submission and generate SQL
@app.route("/generate_sql", methods=["POST"])
def generate():
    xml_input = request.form.get("xml_input", "")
    use_a2a = 'use_a2a' in request.form
    
    if not xml_input:
        return render_template_string(HTML_TEMPLATE, error_message="Please provide valid XML input.")

    if use_a2a:
        # Use the Agent-to-Agent Protocol
        json_plan, error = generate_plan_from_xml(xml_input)
        if error:
            return render_template_string(HTML_TEMPLATE, error_message=error)

        sql_query, error = generate_sql_from_plan(json_plan)
        if error:
            return render_template_string(HTML_TEMPLATE, error_message=error)
            
        return render_template_string(HTML_TEMPLATE, sql_query_a2a=sql_query, json_plan=json_plan)
    else:
        # Use the direct generation method
        sql_query, error = generate_bigquery_sql_from_xml(xml_input)
        if error:
            return render_template_string(HTML_TEMPLATE, error_message=error)
            
        return render_template_string(HTML_TEMPLATE, sql_query_direct=sql_query)

if __name__ == "__main__":
    # The host='0.0.0.0' makes the server externally visible
    # debug=True allows for auto-reloading during development
    app.run(debug=True, host='0.0.0.0', port=5000)
