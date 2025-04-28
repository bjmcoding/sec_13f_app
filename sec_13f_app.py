# requirements.txt:
# streamlit
# requests
# pandas
# lxml
# beautifulsoup4
# ratelimiter
# cachetools # Streamlit now has built-in caching, but cachetools can be useful too

import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
from lxml import etree # Use lxml for robust parsing
from bs4 import BeautifulSoup
import re
import time
from io import StringIO, BytesIO
import logging
from datetime import datetime, timedelta

# --- Configuration ---
USER_AGENT_NAME = "BJMCODING"
USER_AGENT_EMAIL = "bmccormick.seo@gmail.com"
SEC_USER_AGENT = f"{USER_AGENT_NAME} {USER_AGENT_EMAIL}"
# SEC Rate Limit: 10 requests per second
# Add a small buffer, e.g., 1 request every 0.11 seconds
# RateLimiter(max_calls=1, period=0.11) # Alternative: time.sleep(0.11)
SEC_REQUEST_DELAY = 0.11 # seconds between requests

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- SEC EDGAR Interaction Class ---

class SECEdgarHandler:
    """
    Handles interactions with the SEC EDGAR database, including fetching filings,
    finding the 13F Information Table XML, and parsing holdings.
    Adheres to SEC fair access policies.
    """
    BASE_URL = "https://www.sec.gov"
    ARCHIVES_URL = f"{BASE_URL}/Archives/edgar/data"
    SUBMISSIONS_API_URL = f"{BASE_URL}/submissions/CIK{{cik}}.json"

    def __init__(self, user_agent):
        self.user_agent = user_agent
        self.session = self._create_session()

    def _create_session(self):
        """Creates a requests session with retry logic."""
        session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=1, # E.g., 1s, 2s, 4s, 8s, 16s
            status_forcelist=[429, 500, 502, 503, 504], # Retry on these codes
            allowed_methods=["HEAD", "GET", "OPTIONS"] # Python 3.10 deprecated 'method_whitelist'
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        session.headers.update({"User-Agent": self.user_agent})
        return session

    def _make_request(self, url, stream=False):
        """Makes a rate-limited request using the configured session."""
        time.sleep(SEC_REQUEST_DELAY) # Simple rate limiting
        try:
            response = self.session.get(url, stream=stream)
            response.raise_for_status() # Raises HTTPError for bad responses (4XX, 5XX)
            logging.info(f"Successfully fetched: {url} (Status: {response.status_code})")
            return response
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed for {url}: {e}")
            st.error(f"Error fetching data from SEC EDGAR: {url}. Reason: {e}. Check CIK/Accession# and SEC status.")
            return None

    def get_submissions(self, cik):
        """Fetches submission metadata for a given CIK."""
        # Pad CIK to 10 digits
        cik_padded = str(cik).zfill(10)
        url = self.SUBMISSIONS_API_URL.format(cik=cik_padded)
        response = self._make_request(url)
        if response:
            try:
                return response.json()
            except requests.exceptions.JSONDecodeError as e:
                logging.error(f"Failed to decode JSON from {url}: {e}")
                st.error(f"Failed to parse submission data for CIK {cik}.")
                return None
        return None

    def find_latest_13f_hr_accession(self, cik, report_year, report_quarter):
        """
        Finds the accession number for the latest 13F-HR filing for a specific
        CIK and reporting period (year/quarter).
        """
        submissions_data = self.get_submissions(cik)
        if not submissions_data or 'filings' not in submissions_data or 'recent' not in submissions_data['filings']:
            st.warning(f"No recent filings found for CIK {cik}.")
            return None

        target_report_date_str = self._get_quarter_end_date(report_year, report_quarter)
        if not target_report_date_str:
            return None

        # Look through recent filings for a 13F-HR matching the report date
        filings = submissions_data['filings']['recent']
        latest_matching_filing = None
        latest_filing_date = datetime.min

        for i in range(len(filings['accessionNumber'])):
            form = filings['form'][i]
            # We want 13F-HR (Holdings Report) or 13F-HR/A (Amendment)
            if form in ["13F-HR", "13F-HR/A"]:
                report_date = filings['reportDate'][i]
                filing_date_str = filings['filingDate'][i]

                if report_date == target_report_date_str:
                    try:
                        current_filing_date = datetime.strptime(filing_date_str, '%Y-%m-%d')
                        # Get the *latest filed* report for that quarter end
                        if latest_matching_filing is None or current_filing_date > latest_filing_date:
                             latest_filing_date = current_filing_date
                             latest_matching_filing = filings['accessionNumber'][i]
                    except ValueError:
                         logging.warning(f"Could not parse filing date {filing_date_str} for CIK {cik}")
                         continue # Skip if date parsing fails

        if latest_matching_filing:
            logging.info(f"Found accession {latest_matching_filing} for CIK {cik}, Period {target_report_date_str}")
            return latest_matching_filing
        else:
            st.warning(f"No 13F-HR filing found for CIK {cik} for quarter ending {target_report_date_str}.")
            return None

    def _get_quarter_end_date(self, year, quarter):
        """Calculates the quarter-end date string (YYYY-MM-DD)."""
        if quarter == 1:
            return f"{year}-03-31"
        elif quarter == 2:
            return f"{year}-06-30"
        elif quarter == 3:
            return f"{year}-09-30"
        elif quarter == 4:
            return f"{year}-12-31"
        else:
            st.error("Invalid quarter selected.")
            return None

    def _get_primary_doc_url(self, cik, accession_number):
        """Constructs the URL for the primary submission document (.txt)."""
        cik_padded = str(cik).zfill(10)
        accession_no_dash = accession_number.replace('-', '')
        return f"{self.ARCHIVES_URL}/{cik_padded}/{accession_no_dash}/{accession_number}.txt"

    def _get_html_index_url(self, cik, accession_number):
        """Constructs the URL for the HTML index page."""
        cik_padded = str(cik).zfill(10)
        accession_no_dash = accession_number.replace('-', '')
        return f"{self.ARCHIVES_URL}/{cik_padded}/{accession_no_dash}/{accession_number}-index.htm" # Note: sometimes .html

    def _fetch_primary_doc(self, cik, accession_number):
        """Fetches the content of the primary submission document."""
        url = self._get_primary_doc_url(cik, accession_number)
        response = self._make_request(url)
        return response.text if response else None

    def _fetch_html_index(self, cik, accession_number):
        """Fetches the content of the HTML index page."""
        url = self._get_html_index_url(cik, accession_number)
        # Try .htm first, then .html as a fallback
        response = self._make_request(url)
        if response:
            return response.text
        else:
            # Try .html suffix
            url_html = url.replace("-index.htm", "-index.html")
            response_html = self._make_request(url_html)
            return response_html.text if response_html else None


    @st.cache_data(ttl=3600) # Cache results for 1 hour
    def find_info_table_xml_filename(_self, cik, accession_number):
        """
        Robustly finds the Information Table XML filename using the recommended strategy.
        Uses _self because Streamlit caching decorators modify the 'self' argument handling.
        """
        logging.info(f"Searching for Info Table XML for {cik} / {accession_number}")

        # --- Step 1 & 2: Fetch Primary Document ---
        primary_doc_content = _self._fetch_primary_doc(cik, accession_number)
        if not primary_doc_content:
            return None # Error already logged and shown by _fetch_primary_doc

        # --- Step 3: Parse Primary Document Manifest ---
        try:
            # Regex to find <DOCUMENT> blocks, tolerant to variations
            document_blocks = re.findall(r'<DOCUMENT>(.*?)</DOCUMENT>', primary_doc_content, re.DOTALL | re.IGNORECASE)
            logging.info(f"Found {len(document_blocks)} <DOCUMENT> blocks in primary doc.")

            for block in document_blocks:
                type_match = re.search(r'<TYPE>(.*?)\n', block, re.IGNORECASE)
                filename_match = re.search(r'<FILENAME>(.*?)\n', block, re.IGNORECASE)

                if type_match and filename_match:
                    doc_type = type_match.group(1).strip().upper()
                    doc_filename = filename_match.group(1).strip()

                    # Check if the type is INFORMATION TABLE (or common variations)
                    if 'INFORMATION TABLE' in doc_type or 'FORM 13F INFORMATION TABLE' in doc_type:
                         # Ensure it's likely an XML file
                        if doc_filename.lower().endswith('.xml'):
                            logging.info(f"Found Info Table via Manifest: Type='{doc_type}', Filename='{doc_filename}'")
                            return doc_filename
                        else:
                             logging.warning(f"Found INFORMATION TABLE type, but filename '{doc_filename}' is not XML. Skipping.")

        except Exception as e:
            logging.error(f"Error parsing primary document manifest for {accession_number}: {e}")
            # Continue to fallback

        # --- Step 4: Fallback - Parse HTML Index Page ---
        logging.info(f"Manifest parsing failed or yielded no XML. Trying HTML index page for {accession_number}.")
        html_index_content = _self._fetch_html_index(cik, accession_number)
        if html_index_content:
            try:
                soup = BeautifulSoup(html_index_content, 'lxml') # Use lxml parser
                # Find the table listing documents (look for common summary attributes or table structure)
                doc_table = None
                for table in soup.find_all('table'):
                    summary = table.get('summary', '').lower()
                    if 'document format files' in summary or 'submission documents' in summary:
                        doc_table = table
                        break
                    # Fallback: check table headers if summary is missing/unhelpful
                    elif not doc_table:
                         headers = [th.get_text(strip=True).lower() for th in table.find_all('th')]
                         if 'document' in headers and 'type' in headers:
                              doc_table = table
                              # No break here, summary is preferred

                if doc_table:
                    logging.info("Found potential document table in HTML index.")
                    rows = doc_table.find_all('tr')
                    for row in rows[1:]: # Skip header row
                        cells = row.find_all('td')
                        if len(cells) >= 3: # Need at least Seq, Document, Type/Description
                            # Try to find Type/Description cell content reliably
                            doc_link_cell = cells[1] # Usually the second cell has the link
                            type_cell = cells[3] if len(cells) > 3 else cells[2] # Type/Description often 3rd or 4th

                            type_text = type_cell.get_text(strip=True).upper()
                            doc_link = doc_link_cell.find('a')

                            if doc_link and doc_link.has_attr('href'):
                                filename = doc_link.get_text(strip=True)
                                # Check if type indicates information table
                                if 'INFORMATION TABLE' in type_text or '13F TABLE' in type_text:
                                     if filename.lower().endswith('.xml'):
                                        logging.info(f"Found Info Table via HTML Index: Type='{type_text}', Filename='{filename}'")
                                        return filename
                                     else:
                                        logging.warning(f"Found INFORMATION TABLE type in HTML index, but filename '{filename}' is not XML. Skipping.")
            except Exception as e:
                logging.error(f"Error parsing HTML index page for {accession_number}: {e}")
                # Continue to next fallback

        # --- Step 5: Fallback - Check Common Default Filenames (Least Reliable) ---
        logging.warning(f"HTML index parsing failed or yielded no XML. Trying default filenames for {accession_number}.")
        cik_padded = str(cik).zfill(10)
        accession_no_dash = accession_number.replace('-', '')
        base_path = f"{self.ARCHIVES_URL}/{cik_padded}/{accession_no_dash}"
        default_filenames = ['infotable.xml', 'form13fInfoTable.xml', 'information_table.xml'] # Add others if known

        for filename in default_filenames:
            url = f"{base_path}/{filename}"
            time.sleep(SEC_REQUEST_DELAY) # Still need rate limiting
            try:
                # Use HEAD request to check existence without downloading fully
                response = self.session.head(url, timeout=5)
                if response.status_code == 200:
                     logging.info(f"Found Info Table via Default Filename Check: '{filename}'")
                     return filename
            except requests.exceptions.RequestException as e:
                # Log minor errors, as failure here is expected sometimes
                logging.debug(f"HEAD request failed for default {filename} for {accession_number}: {e}")
                continue # Try next default filename

        logging.error(f"Could not identify Information Table XML filename for {accession_number} using any method.")
        st.error(f"Failed to find the Information Table XML file within filing {accession_number}. The filing might be structured unusually, be very old, or missing the XML table.")
        return None

    def _get_info_table_xml_url(self, cik, accession_number, xml_filename):
        """Constructs the full URL to the identified Information Table XML."""
        cik_padded = str(cik).zfill(10)
        accession_no_dash = accession_number.replace('-', '')
        return f"{self.ARCHIVES_URL}/{cik_padded}/{accession_no_dash}/{xml_filename}"

    def _fetch_info_table_xml(self, xml_url):
        """Fetches the content of the Information Table XML."""
        response = self._make_request(xml_url)
        # Use response.content for potentially binary XML data
        return response.content if response else None

    def parse_info_table_xml(self, xml_content):
        """Parses the 13F Information Table XML into a pandas DataFrame."""
        if not xml_content:
            return pd.DataFrame() # Return empty DataFrame if no content

        try:
            # Use BytesIO to handle the XML content (which might be bytes)
            xml_file = BytesIO(xml_content)
            tree = etree.parse(xml_file)
            root = tree.getroot()

            # Define XML namespaces (often present in 13F XML)
            # These might vary slightly, common ones are included
            ns = {
                'ns': 'http://www.sec.gov/edgar/document/thirteenf/informationtable'
                 # Add other potential namespaces if observed in filings
            }
            # Helper to find elements with namespace handling
            def find_all_with_ns(element, path):
                 # Try with default namespace first
                found = element.xpath(f".//ns:{path}", namespaces=ns)
                 # If not found, try without namespace (older filings might lack it)
                if not found:
                     found = element.xpath(f".//{path}")
                return found

            # Extract data for each holding ('infoTable' element)
            holdings = []
            info_tables = find_all_with_ns(root, 'infoTable') # Common parent element for each holding

            for table in info_tables:
                holding = {}
                # Extract fields - use .text attribute and handle missing elements gracefully
                holding['nameOfIssuer'] = find_all_with_ns(table, 'nameOfIssuer')[0].text if find_all_with_ns(table, 'nameOfIssuer') else None
                holding['titleOfClass'] = find_all_with_ns(table, 'titleOfClass')[0].text if find_all_with_ns(table, 'titleOfClass') else None
                holding['cusip'] = find_all_with_ns(table, 'cusip')[0].text if find_all_with_ns(table, 'cusip') else None
                # Value is reported in thousands, multiply by 1000
                value_text = find_all_with_ns(table, 'value')[0].text if find_all_with_ns(table, 'value') else '0'
                holding['value'] = int(value_text.replace(',', '')) * 1000 if value_text else 0
                # Shares or Principal Amount Data
                ssh_prnamt_elem = find_all_with_ns(table, 'shrsOrPrnAmt')
                if ssh_prnamt_elem:
                     holding['sshPrnamt'] = int(ssh_prnamt_elem[0].xpath('.//ns:sshPrnamt', namespaces=ns)[0].text.replace(',', '')) if ssh_prnamt_elem[0].xpath('.//ns:sshPrnamt', namespaces=ns) else 0
                     holding['sshPrnamtType'] = ssh_prnamt_elem[0].xpath('.//ns:sshPrnamtType', namespaces=ns)[0].text if ssh_prnamt_elem[0].xpath('.//ns:sshPrnamtType', namespaces=ns) else None
                else: # Try older structure without sub-elements
                     holding['sshPrnamt'] = int(find_all_with_ns(table, 'sshPrnamt')[0].text.replace(',', '')) if find_all_with_ns(table, 'sshPrnamt') else 0
                     holding['sshPrnamtType'] = find_all_with_ns(table, 'sshPrnamtType')[0].text if find_all_with_ns(table, 'sshPrnamtType') else None

                holding['investmentDiscretion'] = find_all_with_ns(table, 'investmentDiscretion')[0].text if find_all_with_ns(table, 'investmentDiscretion') else None
                # Voting Authority
                voting_auth_elem = find_all_with_ns(table, 'votingAuthority')
                if voting_auth_elem:
                    holding['votingAuthSole'] = int(voting_auth_elem[0].xpath('.//ns:Sole', namespaces=ns)[0].text.replace(',', '')) if voting_auth_elem[0].xpath('.//ns:Sole', namespaces=ns) else 0
                    holding['votingAuthShared'] = int(voting_auth_elem[0].xpath('.//ns:Shared', namespaces=ns)[0].text.replace(',', '')) if voting_auth_elem[0].xpath('.//ns:Shared', namespaces=ns) else 0
                    holding['votingAuthNone'] = int(voting_auth_elem[0].xpath('.//ns:None', namespaces=ns)[0].text.replace(',', '')) if voting_auth_elem[0].xpath('.//ns:None', namespaces=ns) else 0
                else:
                     # Handle cases where voting authority might be missing or structured differently
                     holding['votingAuthSole'] = 0
                     holding['votingAuthShared'] = 0
                     holding['votingAuthNone'] = 0


                holdings.append(holding)

            df = pd.DataFrame(holdings)
            # Basic cleaning
            df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(0).astype(float) # Use float for value
            df['sshPrnamt'] = pd.to_numeric(df['sshPrnamt'], errors='coerce').fillna(0).astype(int) # Shares usually integer
            df['votingAuthSole'] = pd.to_numeric(df['votingAuthSole'], errors='coerce').fillna(0).astype(int)
            df['votingAuthShared'] = pd.to_numeric(df['votingAuthShared'], errors='coerce').fillna(0).astype(int)
            df['votingAuthNone'] = pd.to_numeric(df['votingAuthNone'], errors='coerce').fillna(0).astype(int)
            df['cusip'] = df['cusip'].str.strip().str.upper() # Standardize CUSIP

            logging.info(f"Successfully parsed {len(df)} holdings from XML.")
            return df

        except etree.XMLSyntaxError as e:
            logging.error(f"XML Syntax Error parsing info table: {e}")
            st.error(f"Failed to parse the XML holdings table. It might be malformed. Error: {e}")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"Unexpected error parsing info table XML: {e}")
            st.error(f"An unexpected error occurred during XML parsing: {e}")
            return pd.DataFrame()

# --- Data Processing Functions ---

@st.cache_data(ttl=3600) # Cache results for 1 hour
def get_holdings_data(cik, year, quarter):
    """
    Orchestrates fetching and parsing 13F holdings data for a specific CIK and quarter.
    """
    handler = SECEdgarHandler(SEC_USER_AGENT)
    accession_number = handler.find_latest_13f_hr_accession(cik, year, quarter)
    if not accession_number:
        return pd.DataFrame() # Error/warning shown by find_latest_13f_hr_accession

    xml_filename = handler.find_info_table_xml_filename(cik, accession_number)
    if not xml_filename:
         return pd.DataFrame() # Error shown by find_info_table_xml_filename

    xml_url = handler._get_info_table_xml_url(cik, accession_number, xml_filename)
    xml_content = handler._fetch_info_table_xml(xml_url)
    if not xml_content:
         st.error(f"Failed to download the XML file: {xml_url}")
         return pd.DataFrame()

    return handler.parse_info_table_xml(xml_content)

def get_previous_quarter(year, quarter):
    """ Calculates the year and quarter for the previous period. """
    if quarter == 1:
        return year - 1, 4
    else:
        return year, quarter - 1

@st.cache_data(ttl=3600)
def calculate_holding_changes(_df_curr, _df_prev):
    """
    Compares holdings between two quarters (DataFrames) and calculates changes.
    Uses _df_curr, _df_prev because Streamlit caching decorators modify argument handling.
    """
    if _df_prev.empty:
        # If previous quarter data is missing, mark all current holdings as New
        _df_curr['change_type'] = 'New'
        _df_curr['change_shares'] = _df_curr['sshPrnamt']
        _df_curr['change_value'] = _df_curr['value']
        _df_curr['change_pct'] = 100.0 # Or pd.NA
        return _df_curr[['cusip', 'nameOfIssuer', 'sshPrnamt', 'value', 'change_type', 'change_shares', 'change_value', 'change_pct']]

    # Merge based on CUSIP
    df_merged = pd.merge(
        _df_curr,
        _df_prev,
        on='cusip',
        how='outer',
        suffixes=('_curr', '_prev')
    )

    changes = []
    for _, row in df_merged.iterrows():
        cusip = row['cusip']
        name = row['nameOfIssuer_curr'] if pd.notna(row['nameOfIssuer_curr']) else row['nameOfIssuer_prev']
        shares_curr = row['sshPrnamt_curr'] if pd.notna(row['sshPrnamt_curr']) else 0
        value_curr = row['value_curr'] if pd.notna(row['value_curr']) else 0
        shares_prev = row['sshPrnamt_prev'] if pd.notna(row['sshPrnamt_prev']) else 0
        value_prev = row['value_prev'] if pd.notna(row['value_prev']) else 0

        change_type = ''
        change_shares = 0
        change_value = 0.0
        change_pct = 0.0

        if shares_curr > 0 and shares_prev == 0:
            change_type = 'New'
            change_shares = shares_curr
            change_value = value_curr
            change_pct = 100.0 # Represents a new position
        elif shares_curr == 0 and shares_prev > 0:
            change_type = 'Exited'
            change_shares = -shares_prev # Negative indicates exit of prev amount
            change_value = -value_prev
            change_pct = -100.0 # Represents a full exit
        elif shares_curr > 0 and shares_prev > 0:
            change_shares = shares_curr - shares_prev
            change_value = value_curr - value_prev
            if shares_prev > 0: # Avoid division by zero
                 change_pct = (shares_curr - shares_prev) / shares_prev * 100.0
            else: # Should not happen if shares_prev > 0, but safety check
                 change_pct = 100.0 if shares_curr > 0 else 0.0 # Or pd.NA

            if change_shares > 0:
                change_type = 'Increased'
            elif change_shares < 0:
                change_type = 'Decreased'
            else:
                change_type = 'Unchanged' # Or handle minor value changes if desired
        else:
             # Both are zero or NaN - skip or mark as unchanged/error
             change_type = 'N/A' # Or None
             continue # Don't add rows where nothing was held in either period


        changes.append({
            'cusip': cusip,
            'nameOfIssuer': name,
            'sshPrnamt': shares_curr, # Show current shares
            'value': value_curr,      # Show current value
            'change_type': change_type,
            'change_shares': change_shares,
            'change_value': change_value,
            'change_pct': change_pct
        })

    df_changes = pd.DataFrame(changes)
    return df_changes


# --- Streamlit App UI ---

st.set_page_config(layout="wide", page_title="13F Holdings Tracker")

st.title("📈 SEC Form 13F Institutional Holdings Tracker")
st.caption(f"Using SEC EDGAR data | User Agent: {SEC_USER_AGENT}")

st.sidebar.header("Query Parameters")

# Input CIKs
ciks_input = st.sidebar.text_area(
    "Enter Manager CIKs (comma-separated)",
    "789019, 1067983", # Example: Berkshire Hathaway, Tiger Global Management
    help="Enter the Central Index Key (CIK) for each manager. Find CIKs on the SEC website."
)
ciks = [cik.strip() for cik in ciks_input.split(',') if cik.strip().isdigit()]

# Select Quarter
current_year = datetime.now().year
years = list(range(current_year, current_year - 10, -1)) # Last 10 years
quarters = [1, 2, 3, 4]

selected_year = st.sidebar.selectbox("Year", years)
# Default to Q4 of previous year if current date is before first filing deadline (Feb 15)
# Or default to most recently completed quarter
today = datetime.now().date()
default_q = 4 if today.month < 2 or (today.month == 2 and today.day < 15) else \
            1 if today.month < 5 or (today.month == 5 and today.day < 16) else \
            2 if today.month < 8 or (today.month == 8 and today.day < 15) else \
            3 if today.month < 11 or (today.month == 11 and today.day < 15) else 4
default_year = selected_year -1 if default_q == 4 and today.month < 2 else selected_year # Adjust year if default q is 4

selected_quarter = st.sidebar.selectbox("Quarter", quarters, index=default_q-1)

# Select View
view_options = ["Manager View", "Stock View"]
selected_view = st.sidebar.radio("Select View", view_options)

# Input for Stock View
stock_cusip_input = ""
if selected_view == "Stock View":
    stock_cusip_input = st.sidebar.text_input(
        "Enter Stock CUSIP",
        "", # Example: Apple CUSIP "037833100"
        help="Enter the 9-character CUSIP of the stock to analyze."
    ).strip().upper()

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Limitations:**
    * Data is lagged up to 45 days after quarter end.
    * Only includes specific 'Section 13(f)' securities (mostly US stocks/ETFs).
    * Excludes mutual funds, most bonds, shorts, international stocks (non-US listed).
    * CIKs must be entered manually.
    * Stock view searches only within the managers listed above.
    * Data accuracy depends on filer submissions. Values in $USD.
    """
)

# --- Main Application Logic ---

# Data cache dictionary
if 'holdings_cache' not in st.session_state:
    st.session_state.holdings_cache = {}

def load_manager_data(cik_list, year, quarter):
    """Loads current and previous quarter data for multiple managers into the cache."""
    prev_year, prev_quarter = get_previous_quarter(year, quarter)
    loaded_data = {} # {cik: {'current': df, 'previous': df}}

    for cik in cik_list:
        st.write(f"--- Processing Manager CIK: {cik} ---")
        # Define cache keys
        cache_key_curr = f"{cik}_{year}_Q{quarter}"
        cache_key_prev = f"{cik}_{prev_year}_Q{prev_quarter}"

        with st.spinner(f"Fetching data for {cik} - {year} Q{quarter}..."):
            if cache_key_curr not in st.session_state.holdings_cache:
                 df_curr = get_holdings_data(cik, year, quarter)
                 st.session_state.holdings_cache[cache_key_curr] = df_curr.copy() # Store copy
            else:
                 df_curr = st.session_state.holdings_cache[cache_key_curr]
                 st.info(f"Using cached data for {cik} - {year} Q{quarter}")

        with st.spinner(f"Fetching data for {cik} - {prev_year} Q{prev_quarter} (for comparison)..."):
             if cache_key_prev not in st.session_state.holdings_cache:
                  df_prev = get_holdings_data(cik, prev_year, prev_quarter)
                  st.session_state.holdings_cache[cache_key_prev] = df_prev.copy() # Store copy
             else:
                  df_prev = st.session_state.holdings_cache[cache_key_prev]
                  st.info(f"Using cached data for {cik} - {prev_year} Q{prev_quarter}")

        loaded_data[cik] = {'current': df_curr, 'previous': df_prev}

    return loaded_data

# --- Display Logic ---
if not ciks:
    st.warning("Please enter at least one valid CIK in the sidebar.")
else:
    # Load data for all selected CIKs first
    all_manager_data = load_manager_data(ciks, selected_year, selected_quarter)

    if selected_view == "Manager View":
        st.header(f"Manager View - {selected_year} Q{quarter}")
        selected_cik = st.selectbox("Select Manager CIK to Display", ciks)

        if selected_cik in all_manager_data:
            data = all_manager_data[selected_cik]
            df_curr = data['current']
            df_prev = data['previous']

            if not df_curr.empty:
                st.subheader(f"Holdings Changes for CIK: {selected_cik} (Q{selected_quarter} {selected_year} vs Previous)")
                df_changes = calculate_holding_changes(df_curr, df_prev)

                if not df_changes.empty:
                     # Display summary of changes
                     st.metric("Total Holdings Reported", len(df_curr))
                     change_counts = df_changes['change_type'].value_counts()
                     col1, col2, col3, col4 = st.columns(4)
                     col1.metric("New Positions", change_counts.get('New', 0))
                     col2.metric("Increased Positions", change_counts.get('Increased', 0))
                     col3.metric("Decreased Positions", change_counts.get('Decreased', 0))
                     col4.metric("Exited Positions", change_counts.get('Exited', 0))


                     # Format columns for display
                     df_display = df_changes[[
                         'cusip', 'nameOfIssuer', 'change_type',
                         'sshPrnamt', 'value',
                         'change_shares', 'change_value', 'change_pct'
                     ]].sort_values(by=['change_type', 'value'], ascending=[True, False]) # Sort for clarity

                     # Improve formatting for display
                     st.dataframe(df_display.style.format({
                         'value': "${:,.0f}",
                         'change_shares': "{:,.0f}",
                         'change_value': "${:,.0f}",
                         'change_pct': "{:.1f}%"
                     }), use_container_width=True)

                else:
                     st.info(f"No changes calculated for CIK {selected_cik}. This might happen if only one quarter's data was available.")
                     st.subheader(f"Current Holdings for CIK: {selected_cik} (as of {selected_year}-Q{selected_quarter} end)")
                     st.dataframe(df_curr.style.format({'value': "${:,.0f}"}), use_container_width=True)

            else:
                st.warning(f"No holdings data found or parsed for CIK {selected_cik} for the selected quarter ({selected_year} Q{selected_quarter}).")

    elif selected_view == "Stock View":
        st.header(f"Stock View - {selected_year} Q{quarter}")

        if not stock_cusip_input or len(stock_cusip_input) != 9:
            st.warning("Please enter a valid 9-character CUSIP in the sidebar for the Stock View.")
        elif not all_manager_data:
             st.warning("No manager data loaded. Please check CIKs and selected quarter.")
        else:
            st.subheader(f"Activity in Stock CUSIP: {stock_cusip_input} by Selected Managers")
            stock_activity = []

            for cik, data in all_manager_data.items():
                 df_curr = data['current']
                 df_prev = data['previous']

                 # Calculate changes specific to this manager for context
                 df_changes = calculate_holding_changes(df_curr, df_prev)

                 # Find the specific stock in the changes df
                 stock_row = df_changes[df_changes['cusip'] == stock_cusip_input]

                 if not stock_row.empty:
                     activity = stock_row.iloc[0].to_dict()
                     activity['cik'] = cik
                     stock_activity.append(activity)
                 else:
                      # Check if it existed previously but was exited (might be missed by change calc if curr is empty)
                      if not df_prev.empty:
                           prev_row = df_prev[df_prev['cusip'] == stock_cusip_input]
                           if not prev_row.empty:
                                stock_activity.append({
                                     'cik': cik,
                                     'cusip': stock_cusip_input,
                                     'nameOfIssuer': prev_row.iloc[0]['nameOfIssuer'],
                                     'sshPrnamt': 0, # Current shares are 0
                                     'value': 0,     # Current value is 0
                                     'change_type': 'Exited',
                                     'change_shares': -prev_row.iloc[0]['sshPrnamt'],
                                     'change_value': -prev_row.iloc[0]['value'],
                                     'change_pct': -100.0
                                })


            if stock_activity:
                df_stock_view = pd.DataFrame(stock_activity)
                # Reorder and format columns
                df_stock_display = df_stock_view[[
                     'cik', 'cusip', 'nameOfIssuer', 'change_type',
                     'sshPrnamt', 'value',
                     'change_shares', 'change_value', 'change_pct'
                ]].sort_values(by=['change_type', 'value'], ascending=[True, False])

                st.dataframe(df_stock_display.style.format({
                     'value': "${:,.0f}",
                     'change_shares': "{:,.0f}",
                     'change_value': "${:,.0f}",
                     'change_pct': "{:.1f}%"
                }), use_container_width=True)
            else:
                st.info(f"No activity found for CUSIP {stock_cusip_input} among the selected managers ({', '.join(ciks)}) for the period {selected_year} Q{quarter} vs Previous.")
