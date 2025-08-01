import os
import json
import xml.etree.ElementTree as ET
from flask import Flask, request, render_template_string
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel

# Initialize the Flask application
app = Flask(__name__)

# --- Configuration for Vertex AI ---
# The VM's service account credentials will be used for authentication.
# Update 'your-gcp-project-id' with your actual project ID.
PROJECT_ID = "your-gcp-project-id"

# Replace this with a region where the Gemini model is enabled for your project.
REGION = "us-central1"

# The model name for Gemini 2.0
MODEL_NAME = "gemini-2.0"

# Initialize the Vertex AI client
aiplatform.init(project=PROJECT_ID, location=REGION)

def parse_alteryx_xml(xml_string):
    """
    Parses a simplified Alteryx XML string to extract a list of tools.
    This function currently supports 'Select' and 'Filter' tools.
    """
    try:
        root = ET.fromstring(xml_string)
        tools = []
        
        # Iterate through all 'Node' elements in the XML
        for node in root.findall('.//Node'):
            tool_id = node.get('ToolID')
            tool_type = node.get('Type')
            
            if tool_type == 'Select':
                fields_element = node.find('.//Fields')
                fields = []
                if fields_element is not None:
                    for field in fields_element.findall('Field'):
                        fields.append({
                            'name': field.get('Name'),
                            'selected': field.get('Selected') == 'True',
                            'rename': field.get('Rename')
                        })
                tools.append({'id': tool_id, 'type': 'Select', 'fields': fields, 'xml_snippet': ET.tostring(node, encoding='unicode')})
            
            elif tool_type == 'Filter':
                expression_element = node.find('.//Expression')
                expression = expression_element.text if expression_element is not None else ""
                tools.append({'id': tool_id, 'type': 'Filter', 'expression': expression, 'xml_snippet': ET.tostring(node, encoding='unicode')})
        
        # Sort tools by their ID to maintain workflow order
        tools.sort(key=lambda x: int(x['id']))
        return tools, None
    
    except ET.ParseError as e:
        return None, f"Error parsing XML: {e}"
    except Exception as e:
        return None, f"An unexpected error occurred during XML parsing: {e}"

def generate_sql_from_tools(tools):
    """
    Iterates through the list of Alteryx tools, calls the Gemini model for each,
    and assembles a final BigQuery SQL query with CTEs.
    """
    try:
        model = GenerativeModel(MODEL_NAME)
        sql_steps = []
        # Initial schema for the source data. This is a simplification; a real app would infer this.
        current_schema = {
            "OrderID": "STRING", "CustomerName": "STRING", 
            "ProductCategory": "STRING", "SalesAmount": "FLOAT"
        }
        current_cte = "initial_data"

        for i, tool in enumerate(tools):
            prompt = ""
            output_schema = current_schema.copy() # Start with a copy of the current schema
            next_cte = f"cte_{i+1}"

            if tool['type'] == 'Select':
                # Update schema based on the Select tool's configuration
                output_schema = {
                    (field['rename'] if field['rename'] else field['name']): current_schema[field['name']]
                    for field in tool['fields'] if field['selected']
                }
                
                prompt = f"""
You are a SQL agent specialized in converting Alteryx workflows to BigQuery SQL.
Your task is to translate the following Alteryx 'Select' tool's logic into a single BigQuery SQL SELECT statement.

- **Input CTE:** `{current_cte}`
- **Input Schema:** {json.dumps(current_schema)}
- **Alteryx Tool XML:**
{tool['xml_snippet']}

Generate only the BigQuery SQL SELECT statement. Do not include any explanations.
"""
            elif tool['type'] == 'Filter':
                # A Filter tool doesn't change the schema, so the output schema is the same.
                prompt = f"""
You are a SQL agent specialized in converting Alteryx workflows to BigQuery SQL.
Your task is to translate the following Alteryx 'Filter' tool's expression into a BigQuery SQL WHERE clause.

- **Input CTE:** `{current_cte}`
- **Input Schema:** {json.dumps(current_schema)}
- **Alteryx Tool XML:**
{tool['xml_snippet']}

Generate only the BigQuery SQL WHERE clause, including the 'WHERE' keyword. Do not include any explanations.
"""
            
            # Use exponential backoff for API calls to handle rate limits
            retries = 0
            max_retries = 5
            while retries < max_retries:
                try:
                    response = model.generate_content(prompt)
                    break
                except Exception as e:
                    retries += 1
                    delay = 2 ** retries
                    print(f"API call failed, retrying in {delay}s...")
                    time.sleep(delay)
            else:
                return None, f"Failed to get a response from the model after {max_retries} retries."

            generated_sql_snippet = response.text.strip().replace("```sql", "").replace("```", "")

            # Build the CTE for the current step
            if tool['type'] == 'Select':
                sql_steps.append(
                    f"WITH {next_cte} AS (\n"
                    f"    {generated_sql_snippet}\n"
                    f"FROM {current_cte}\n"
                    f")"
                )
            elif tool['type'] == 'Filter':
                select_cols = ", ".join(output_schema.keys())
                sql_steps.append(
                    f"WITH {next_cte} AS (\n"
                    f"    SELECT {select_cols}\n"
                    f"    FROM {current_cte}\n"
                    f"    {generated_sql_snippet}\n"
                    f")"
                )

            current_cte = next_cte
            current_schema = output_schema.copy()

        # Assemble the final BigQuery view statement
        if not sql_steps:
            return None, "No SQL steps were generated."
            
        final_sql_body = "\n\n".join(sql_steps)
        final_select_cols = ", ".join(current_schema.keys())
        
        # Replace the initial CTE placeholder with a real table
        final_sql_body = final_sql_body.replace(f"FROM {tools[0]['id']}", "FROM `your_project.your_dataset.your_initial_table`")

        final_query = (
            f"CREATE OR REPLACE VIEW `your_project.your_dataset.your_view_name` AS\n"
            f"{final_sql_body}\n\n"
            f"SELECT\n"
            f"    {final_select_cols}\n"
            f"FROM\n"
            f"    {current_cte};"
        )
        return final_query, None

    except Exception as e:
        return None, f"An error occurred during SQL generation: {e}"


# --- HTML Template for the Web Interface ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Alteryx to BigQuery SQL Converter</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { font-family: 'Inter', sans-serif; }
        textarea { font-family: monospace; }
    </style>
</head>
<body class="bg-gray-100 flex flex-col items-center justify-center min-h-screen p-4">
    <div class="bg-white p-8 rounded-xl shadow-lg w-full max-w-4xl">
        <h1 class="text-3xl font-bold text-center text-gray-800 mb-6">Alteryx to BigQuery SQL Converter</h1>
        <p class="text-center text-gray-600 mb-8">
            Paste a simplified Alteryx workflow XML to generate a BigQuery SQL `CREATE VIEW` statement.
        </p>

        <form action="/generate_sql" method="post" class="space-y-6">
            <div>
                <label for="alteryx_xml_input" class="block text-sm font-medium text-gray-700">Alteryx XML Input</label>
                <textarea id="alteryx_xml_input" name="alteryx_xml_input" rows="10" 
                          class="mt-1 block w-full rounded-md border-gray-300 shadow-sm 
                          focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm p-3 border-2"
                          placeholder="<AlteryxWorkflow>...</AlteryxWorkflow>"><AlteryxWorkflow>
  <Node ToolID="1" Type="Select">
    <Name>Select Columns</Name>
    <Configuration>
      <Fields>
        <Field Name="OrderID" Selected="True" Rename="transaction_id" />
        <Field Name="CustomerName" Selected="True" />
        <Field Name="ProductCategory" Selected="False" />
        <Field Name="SalesAmount" Selected="True" Rename="total_sales" />
      </Fields>
    </Configuration>
  </Node>
  <Node ToolID="2" Type="Filter">
    <Name>Filter High Sales</Name>
    <Configuration>
      <Expression>[total_sales] > 1000 AND [CustomerName] = 'Alice'</Expression>
    </Configuration>
  </Node>
</AlteryxWorkflow></textarea>
            </div>
            
            <div class="flex justify-center">
                <button type="submit" 
                        class="flex justify-center py-2 px-6 border border-transparent 
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
        
        {% if sql_query %}
        <div class="mt-8">
            <h2 class="text-xl font-semibold text-gray-800 mb-3">Generated BigQuery SQL View</h2>
            <div class="bg-gray-800 text-gray-200 rounded-md p-4 overflow-x-auto shadow-inner">
                <pre><code class="whitespace-pre-wrap">{{ sql_query }}</code></pre>
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
    alteryx_xml_input = request.form.get("alteryx_xml_input", "")
    
    if not alteryx_xml_input:
        return render_template_string(HTML_TEMPLATE, error_message="Please provide valid Alteryx XML input.")

    # Parse the Alteryx XML into a list of tools
    tools, error = parse_alteryx_xml(alteryx_xml_input)
    if error:
        return render_template_string(HTML_TEMPLATE, error_message=error)
    
    # Generate the SQL from the parsed tools
    sql_query, error = generate_sql_from_tools(tools)
    if error:
        return render_template_string(HTML_TEMPLATE, error_message=error)
            
    return render_template_string(HTML_TEMPLATE, sql_query=sql_query)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
