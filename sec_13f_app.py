# requirements.txt should contain:
# streamlit>=1.30.0
# requests>=2.28.0
# pandas>=1.5.0
# lxml>=4.9.0
# beautifulsoup4>=4.11.0
# cachetools>=5.0.0

import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
from lxml import etree # Use lxml for robust parsing
from bs4 import BeautifulSoup
import re
import time
# Removed: from ratelimiter import RateLimiter (no longer needed/used)
from io import StringIO, BytesIO
import logging
from datetime import datetime, timedelta

# --- Configuration ---
USER_AGENT_NAME = "BJMCODING"
USER_AGENT_EMAIL = "bmccormick.seo@gmail.com"
SEC_USER_AGENT = f"{USER_AGENT_NAME} {USER_AGENT_EMAIL}"
# SEC Rate Limit: 10 requests per second
# Add a small buffer, e.g., 1 request every 0.11 seconds
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
    # Ensure correct indentation for class variables (Level 1)
    BASE_URL = "https://www.sec.gov"
    ARCHIVES_URL = f"{BASE_URL}/Archives/edgar/data"
    # FIX: Use data.sec.gov for submissions API endpoint
    SUBMISSIONS_API_URL = f"https://data.sec.gov/submissions/CIK{{cik}}.json"

    # Ensure correct indentation for methods (Level 1)
    def __init__(self, user_agent): # <<< Line 47 area - Ensure 4 spaces indentation
        # Ensure correct indentation for code within methods (Level 2)
        self.user_agent = user_agent
        self.session = self._create_session()

    def _create_session(self):
        """Creates a requests session with retry logic."""
        session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=1, # E.g., 1s, 2s, 4s, 8s, 16s
            status_forcelist=[429, 500, 502, 503, 504], # Retry on these codes
            allowed_methods=["HEAD", "GET", "OPTIONS"]
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
            # Log specific error types if possible
            status_code = e.response.status_code if e.response is not None else "N/A"
            logging.error(f"Request failed for {url} (Status: {status_code}): {e}")
            # Provide more context in the Streamlit error
            error_detail = f"Status Code: {status_code}, Reason: {e}"
            st.error(f"Error fetching data from SEC EDGAR: {url}. {error_detail}. Check CIK/Accession# and SEC EDGAR status.")
            return None

    def get_submissions(self, cik):
        """Fetches submission metadata for a given CIK."""
        cik_padded = str(cik).zfill(10)
        url = self.SUBMISSIONS_API_URL.format(cik=cik_padded)
        response = self._make_request(url)
        if response:
            try:
                return response.json()
            except requests.exceptions.JSONDecodeError as e:
                logging.error(f"Failed to decode JSON from {url}: {e}")
                st.error(f"Failed to parse submission data for CIK {cik}. The API might be down or the CIK incorrect.")
                return None
        return None

    def find_latest_13f_hr_accession(self, cik, report_year, report_quarter):
        """
        Finds the accession number for the latest 13F-HR filing for a specific
        CIK and reporting period (year/quarter).
        """
        submissions_data = self.get_submissions(cik)
        # Handle case where submissions_data might be None or missing expected keys
        if not submissions_data:
             st.warning(f"Could not retrieve submission data for CIK {cik}. Check CIK or SEC API status.")
             return None
        if 'filings' not in submissions_data or 'recent' not in submissions_data['filings']:
            st.warning(f"No recent filings structure found for CIK {cik} in API response.")
            # Check if 'filings' exists but is empty
            if 'filings' in submissions_data and not submissions_data['filings'].get('recent'):
                 st.info(f"The 'recent' filings list appears empty for CIK {cik}.")
            return None
        # Further check if required keys exist within 'recent'
        recent_filings = submissions_data['filings']['recent']
        required_keys = ['accessionNumber', 'form', 'reportDate', 'filingDate']
        if not all(key in recent_filings for key in required_keys):
             st.warning(f"Submission data for CIK {cik} is missing expected keys (e.g., accessionNumber, form). API structure might have changed.")
             return None
        # Check if lists are empty
        if not recent_filings['accessionNumber']:
             st.warning(f"No recent accession numbers found for CIK {cik}.")
             return None


        target_report_date_str = self._get_quarter_end_date(report_year, report_quarter)
        if not target_report_date_str:
            return None

        # Look through recent filings for a 13F-HR matching the report date
        filings = submissions_data['filings']['recent']
        latest_matching_filing = None
        latest_filing_date = datetime.min.date() # Compare dates only

        try:
            for i in range(len(filings['accessionNumber'])):
                form = filings['form'][i]
                # We want 13F-HR (Holdings Report) or 13F-HR/A (Amendment)
                if form in ["13F-HR", "13F-HR/A"]:
                    report_date = filings['reportDate'][i]
                    filing_date_str = filings['filingDate'][i]

                    if report_date == target_report_date_str:
                        try:
                            # Parse only the date part
                            current_filing_date = datetime.strptime(filing_date_str, '%Y-%m-%d').date()
                            # Get the *latest filed* report for that quarter end
                            if latest_matching_filing is None or current_filing_date >= latest_filing_date:
                                 latest_filing_date = current_filing_date
                                 latest_matching_filing = filings['accessionNumber'][i]
                        except ValueError:
                             logging.warning(f"Could not parse filing date {filing_date_str} for CIK {cik}")
                             continue # Skip if date parsing fails
        except KeyError as e:
             st.error(f"Missing key '{e}' in filings data for CIK {cik}. API response structure might be unexpected.")
             return None


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
        # Primary name is -index.html, fallback to .htm is handled below
        return f"{self.ARCHIVES_URL}/{cik_padded}/{accession_no_dash}/{accession_number}-index.html"

    def _fetch_primary_doc(self, cik, accession_number):
        """Fetches the content of the primary submission document."""
        url = self._get_primary_doc_url(cik, accession_number)
        response = self._make_request(url)
        return response.text if response else None

    def _fetch_html_index(self, cik, accession_number):
        """Fetches the content of the HTML index page."""
        url_html = self._get_html_index_url(cik, accession_number)
        response = self._make_request(url_html)
        if response:
            return response.text
        else:
            # Fallback: try .htm suffix if .html failed
            url_htm = url_html.replace("-index.html", "-index.htm")
            logging.info(f"Primary index URL failed ({url_html}), trying fallback: {url_htm}")
            response_htm = self._make_request(url_htm)
            return response_htm.text if response_htm else None

    # Use _self convention for methods decorated with st.cache_data
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
            # Error already logged and potentially shown by _fetch_primary_doc
            return None

        # --- Step 3: Parse Primary Document Manifest ---
        try:
            # Regex to find <DOCUMENT> blocks, tolerant to variations
            # Using re.split is sometimes more robust than re.findall for nested/malformed structures
            parts = re.split(r'</?DOCUMENT>', primary_doc_content, flags=re.IGNORECASE | re.DOTALL)
            # Skip the first part (before the first <DOCUMENT>) and process pairs
            document_blocks = parts[1:] # Get content within <DOCUMENT> tags
            logging.info(f"Found {len(document_blocks)} potential document blocks in primary doc.")

            for block in document_blocks:
                # More robust regex to handle potential extra whitespace/newlines
                type_match = re.search(r'<TYPE>\s*(.*?)\s*<', block, re.IGNORECASE | re.DOTALL)
                filename_match = re.search(r'<FILENAME>\s*(.*?)\s*<', block, re.IGNORECASE | re.DOTALL)

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
                doc_table = None
                # Look for tables containing document links
                for table in soup.find_all('table'):
                     # Check for specific attributes or headers
                     summary = table.get('summary', '').lower()
                     if 'document format files' in summary or 'submission documents' in summary:
                          doc_table = table
                          break
                     # Fallback header check
                     headers = [th.get_text(strip=True).lower() for th in table.find_all('th')]
                     if 'document' in headers and ('type' in headers or 'description' in headers):
                          doc_table = table
                          # Don't break here if summary might be better later

                if doc_table:
                    logging.info("Found potential document table in HTML index.")
                    rows = doc_table.find_all('tr')
                    for row in rows[1:]: # Skip header row
                        cells = row.find_all('td')
                        if len(cells) >= 3: # Need at least Seq, Document link, Type/Description
                            try:
                                # Try to get data more reliably by position/content
                                doc_link_cell = cells[1]
                                type_desc_cell = cells[3] if len(cells) > 3 else cells[2] # Usually 3rd or 4th

                                type_desc_text = type_desc_cell.get_text(strip=True).upper()
                                doc_link = doc_link_cell.find('a')

                                if doc_link and doc_link.has_attr('href'):
                                    filename = doc_link.get_text(strip=True)
                                    # Check if type/description indicates information table
                                    if 'INFORMATION TABLE' in type_desc_text or '13F TABLE' in type_desc_text or type_desc_text == 'XML': # Some might just say XML here
                                         if filename.lower().endswith('.xml'):
                                             logging.info(f"Found Info Table via HTML Index: Type='{type_desc_text}', Filename='{filename}'")
                                             return filename
                                         else:
                                             # It might link to HTML version, check href
                                             href = doc_link['href']
                                             if href.lower().endswith('.xml'):
                                                 xml_filename_from_href = href.split('/')[-1]
                                                 logging.info(f"Found Info Table XML via HTML Index Href: Type='{type_desc_text}', Href='{href}', Filename='{xml_filename_from_href}'")
                                                 return xml_filename_from_href

                                            # If filename doesn't end with .xml and href doesn't either, skip
                                             logging.warning(f"Found INFORMATION TABLE type '{type_desc_text}' in HTML index, but link '{filename}'/'{href}' doesn't point to XML. Skipping.")
                            except IndexError:
                                 logging.warning("Skipping row in HTML table due to unexpected cell count.")
                                 continue
            except Exception as e:
                logging.error(f"Error parsing HTML index page for {accession_number}: {e}")
                # Continue to next fallback

        # --- Step 5: Fallback - Check Common Default Filenames (Least Reliable) ---
        logging.warning(f"HTML index parsing failed or yielded no XML. Trying default filenames for {accession_number}.")
        cik_padded = str(cik).zfill(10)
        accession_no_dash = accession_number.replace('-', '')
        base_path = f"{_self.ARCHIVES_URL}/{cik_padded}/{accession_no_dash}" # Use _self inside cached method
        default_filenames = ['infotable.xml', 'form13fInfoTable.xml', 'information_table.xml']

        for filename in default_filenames:
            url = f"{base_path}/{filename}"
            time.sleep(SEC_REQUEST_DELAY) # Still need rate limiting
            try:
                # Use HEAD request to check existence without downloading fully
                response = _self.session.head(url, timeout=5) # Use _self inside cached method
                if response.status_code == 200:
                     # Double check content type if possible
                     content_type = response.headers.get('Content-Type', '').lower()
                     if 'xml' in content_type:
                          logging.info(f"Found Info Table via Default Filename Check: '{filename}'")
                          return filename
                     else:
                          logging.warning(f"Default filename '{filename}' found but Content-Type is '{content_type}'. Skipping.")
            except requests.exceptions.RequestException as e:
                # Log minor errors, as failure here is expected sometimes
                logging.debug(f"HEAD request failed for default {filename} for {accession_number}: {e}")
                continue # Try next default filename

        logging.error(f"Could not identify Information Table XML filename for {accession_number} using any method.")
        st.error(f"Failed to find the Information Table XML file within filing {accession_number}. The filing might be structured unusually, very old, missing the XML table, or access was blocked.")
        return None

    def _get_info_table_xml_url(self, cik, accession_number, xml_filename):
        """Constructs the full URL to the identified Information Table XML."""
        cik_padded = str(cik).zfill(10)
        accession_no_dash = accession_number.replace('-', '')
        # Ensure filename doesn't have leading slashes if extracted from href
        xml_filename_cleaned = xml_filename.lstrip('/')
        return f"{self.ARCHIVES_URL}/{cik_padded}/{accession_no_dash}/{xml_filename_cleaned}"

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
            # Use BytesIO to handle the XML content
            xml_file = BytesIO(xml_content)
            # Try parsing, remove processing instructions which can cause issues
            tree = etree.parse(xml_file, parser=etree.XMLParser(remove_pis=True))
            root = tree.getroot()

            # Define XML namespaces dynamically if possible, or use common ones
            # Attempt to get namespace from root element
            nsmap = root.nsmap
            default_ns_key = None
            # Find the key for the default namespace (if any)
            for key, value in nsmap.items():
                if key is None: # Default namespace has None key in lxml nsmap
                     default_ns_key = "defns" # Assign a temporary key for xpath
                     nsmap[default_ns_key] = value
                     break

            # Fallback/Common namespaces if dynamic detection fails or isn't enough
            ns_common = {
                'ns': 'http://www.sec.gov/edgar/document/thirteenf/informationtable',
                'ns1': 'http://www.sec.gov/edgar/thirteenffiler'
                # Add others if needed based on observation
            }
            # Combine dynamic and common namespaces (dynamic takes precedence if key overlaps)
            ns = ns_common.copy()
            if nsmap:
                 ns.update(nsmap)

            # Helper to find elements with namespace handling
            def find_all_with_ns(element, path_segments):
                xpath_expr = ".//"
                xpath_parts = []
                for segment in path_segments:
                    # Try default namespace first if detected
                    if default_ns_key and default_ns_key in ns:
                         xpath_parts.append(f"{default_ns_key}:{segment}")
                    else:
                         # Try common 'ns' prefix
                         xpath_parts.append(f"ns:{segment}")
                         # Add fallback without namespace
                         xpath_parts.append(f"{segment}")

                # Construct a more flexible XPath: //(defns:tag|ns:tag|tag)
                xpath_expr += "/".join([f"({ '|'.join(xpath_parts[i*len(path_segments):(i+1)*len(path_segments)]) })" for i in range(len(path_segments))])

                # Simpler approach: try combinations known to work
                paths_to_try = []
                base_path = ".//" + "/".join(path_segments)
                paths_to_try.append(base_path) # No namespace
                if default_ns_key:
                     paths_to_try.append(".//" + "/".join([f"{default_ns_key}:{s}" for s in path_segments]))
                paths_to_try.append(".//" + "/".join([f"ns:{s}" for s in path_segments])) # Common ns

                found = []
                for p in paths_to_try:
                    try:
                        found = element.xpath(p, namespaces=ns)
                        if found: break # Stop if found
                    except etree.XPathEvalError: # Handle invalid expressions if namespaces missing
                        continue
                return found

            # Extract data for each holding ('infoTable' element)
            holdings = []
            # Find 'infoTable' elements using flexible namespace search
            info_tables = root.xpath('.//infoTable | .//ns:infoTable | .//defns:infoTable', namespaces=ns)
            if not info_tables:
                logging.warning("Could not find any 'infoTable' elements in the XML.")
                return pd.DataFrame()

            logging.info(f"Found {len(info_tables)} infoTable elements.")

            for table in info_tables:
                 holding = {}
                 # Helper to safely get text from the first element found
                 def get_text_safe(element, path_list):
                      found = find_all_with_ns(element, path_list)
                      return found[0].text.strip() if found and found[0].text else None

                 holding['nameOfIssuer'] = get_text_safe(table, ['nameOfIssuer'])
                 holding['titleOfClass'] = get_text_safe(table, ['titleOfClass'])
                 holding['cusip'] = get_text_safe(table, ['cusip'])

                 value_text = get_text_safe(table, ['value'])
                 holding['value'] = int(value_text.replace(',', '')) * 1000 if value_text else 0

                 ssh_prnamt = get_text_safe(table, ['shrsOrPrnAmt', 'sshPrnamt'])
                 ssh_prnamt_type = get_text_safe(table, ['shrsOrPrnAmt', 'sshPrnamtType'])
                 # Handle cases where shrsOrPrnAmt might be directly under infoTable
                 if ssh_prnamt is None:
                     ssh_prnamt = get_text_safe(table, ['sshPrnamt'])
                 if ssh_prnamt_type is None:
                      ssh_prnamt_type = get_text_safe(table, ['sshPrnamtType'])

                 holding['sshPrnamt'] = int(ssh_prnamt.replace(',', '')) if ssh_prnamt else 0
                 holding['sshPrnamtType'] = ssh_prnamt_type

                 holding['investmentDiscretion'] = get_text_safe(table, ['investmentDiscretion'])

                 sole_auth = get_text_safe(table, ['votingAuthority', 'Sole'])
                 shared_auth = get_text_safe(table, ['votingAuthority', 'Shared'])
                 none_auth = get_text_safe(table, ['votingAuthority', 'None'])

                 holding['votingAuthSole'] = int(sole_auth.replace(',', '')) if sole_auth else 0
                 holding['votingAuthShared'] = int(shared_auth.replace(',', '')) if shared_auth else 0
                 holding['votingAuthNone'] = int(none_auth.replace(',', '')) if none_auth else 0

                 # Add only if CUSIP is present (basic validity check)
                 if holding['cusip']:
                      holdings.append(holding)
                 else:
                      logging.warning(f"Skipping holding record due to missing CUSIP: {holding.get('nameOfIssuer')}")


            if not holdings:
                 logging.warning("No valid holdings extracted from infoTable elements.")
                 return pd.DataFrame()

            df = pd.DataFrame(holdings)
            # Basic cleaning
            df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(0).astype(float)
            df['sshPrnamt'] = pd.to_numeric(df['sshPrnamt'], errors='coerce').fillna(0).astype(int)
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
    # Create handler instance inside function to ensure session is fresh if needed,
    # though caching might make this less critical unless sessions expire/fail.
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
    # Ensure dataframes have necessary columns, handle potential errors if parsing failed partially
    required_cols = ['cusip', 'nameOfIssuer', 'sshPrnamt', 'value']
    if not all(col in _df_curr.columns for col in required_cols):
         logging.error("Current holdings DataFrame is missing required columns for change calculation.")
         # Return minimal change info if possible, or empty
         _df_curr['change_type'] = 'Error'
         _df_curr['change_shares'] = pd.NA
         _df_curr['change_value'] = pd.NA
         _df_curr['change_pct'] = pd.NA
         return _df_curr
    if not _df_prev.empty and not all(col in _df_prev.columns for col in required_cols):
          logging.warning("Previous holdings DataFrame is missing required columns. Treating as empty for change calc.")
          _df_prev = pd.DataFrame(columns=_df_curr.columns) # Use empty df with same columns

    # Make copies to avoid modifying cached dataframes
    df_curr_copy = _df_curr.copy()
    df_prev_copy = _df_prev.copy()

    if df_prev_copy.empty:
        # If previous quarter data is missing, mark all current holdings as New
        df_curr_copy['change_type'] = 'New'
        df_curr_copy['change_shares'] = df_curr_copy['sshPrnamt']
        df_curr_copy['change_value'] = df_curr_copy['value']
        df_curr_copy['change_pct'] = 100.0 # Or pd.NA
        return df_curr_copy[['cusip', 'nameOfIssuer', 'sshPrnamt', 'value', 'change_type', 'change_shares', 'change_value', 'change_pct']]

    # Aggregate duplicate CUSIPs before merging (e.g., separate PUT/CALL options on same CUSIP)
    # Sum numeric columns, keep first for strings
    df_curr_agg = df_curr_copy.groupby('cusip').agg(
        nameOfIssuer=('nameOfIssuer', 'first'),
        sshPrnamt=('sshPrnamt', 'sum'),
        value=('value', 'sum'),
        # Add other columns if needed, e.g., take 'first' titleOfClass
    ).reset_index()
    df_prev_agg = df_prev_copy.groupby('cusip').agg(
        nameOfIssuer=('nameOfIssuer', 'first'),
        sshPrnamt=('sshPrnamt', 'sum'),
        value=('value', 'sum'),
    ).reset_index()


    # Merge based on CUSIP using aggregated data
    df_merged = pd.merge(
        df_curr_agg,
        df_prev_agg,
        on='cusip',
        how='outer',
        suffixes=('_curr', '_prev')
    )

    changes = []
    for _, row in df_merged.iterrows():
        cusip = row['cusip']
        # Use current name if available, otherwise previous name
        name = row['nameOfIssuer_curr'] if pd.notna(row['nameOfIssuer_curr']) else row['nameOfIssuer_prev']
        # Handle potential NaN values from outer join, default to 0
        shares_curr = row['sshPrnamt_curr'] if pd.notna(row['sshPrnamt_curr']) else 0
        value_curr = row['value_curr'] if pd.notna(row['value_curr']) else 0
        shares_prev = row['sshPrnamt_prev'] if pd.notna(row['sshPrnamt_prev']) else 0
        value_prev = row['value_prev'] if pd.notna(row['value_prev']) else 0

        change_type = ''
        change_shares = 0
        change_value = 0.0
        change_pct = 0.0

        # Determine change type
        is_new = shares_curr > 0 and shares_prev == 0
        is_exited = shares_curr == 0 and shares_prev > 0
        is_held_both = shares_curr > 0 and shares_prev > 0

        if is_new:
            change_type = 'New'
            change_shares = shares_curr
            change_value = value_curr
            change_pct = 100.0
        elif is_exited:
            change_type = 'Exited'
            change_shares = -shares_prev
            change_value = -value_prev
            change_pct = -100.0
        elif is_held_both:
            change_shares = shares_curr - shares_prev
            change_value = value_curr - value_prev
            # Calculate percentage change carefully
            if shares_prev != 0:
                 change_pct = (shares_curr - shares_prev) / shares_prev * 100.0
            else: # Should not happen if shares_prev > 0, but safety
                 change_pct = pd.NA # Indeterminate percentage change from zero

            # Assign type based on share change
            if change_shares > 0:
                change_type = 'Increased'
            elif change_shares < 0:
                change_type = 'Decreased'
            else:
                 # If shares are same, check value change (could be price fluctuation)
                 if value_curr != value_prev:
                     change_type = 'Value Change Only'
                 else:
                      change_type = 'Unchanged'
        else:
             # Both are zero - this row resulted from merge but represents no holding
             continue # Skip rows where holding was zero in both periods


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

    if not changes:
        return pd.DataFrame(columns=['cusip', 'nameOfIssuer', 'sshPrnamt', 'value', 'change_type', 'change_shares', 'change_value', 'change_pct'])

    df_changes = pd.DataFrame(changes)
    # Ensure correct types after creation
    df_changes['value'] = pd.to_numeric(df_changes['value'], errors='coerce').fillna(0).astype(float)
    df_changes['sshPrnamt'] = pd.to_numeric(df_changes['sshPrnamt'], errors='coerce').fillna(0).astype(int)
    df_changes['change_shares'] = pd.to_numeric(df_changes['change_shares'], errors='coerce').fillna(0).astype(int)
    df_changes['change_value'] = pd.to_numeric(df_changes['change_value'], errors='coerce').fillna(0).astype(float)
    df_changes['change_pct'] = pd.to_numeric(df_changes['change_pct'], errors='coerce').fillna(pd.NA) # Allow NA for percentage

    return df_changes


# --- Streamlit App UI ---

st.set_page_config(layout="wide", page_title="13F Holdings Tracker")

st.title("📈 SEC Form 13F Institutional Holdings Tracker")
st.caption(f"Using SEC EDGAR data | User Agent: {SEC_USER_AGENT}")

st.sidebar.header("Query Parameters")

# Input CIKs
# Increased examples: Added Scion Asset Management, Pershing Square
ciks_input = st.sidebar.text_area(
    "Enter Manager CIKs (comma-separated)",
    "789019, 1067983, 1649339, 1336528", # Example: BRK, Tiger, Scion, Pershing
    help="Enter the Central Index Key (CIK) for each manager. Find CIKs on the SEC website."
)
# Validate CIKs are numeric and add to list
ciks = []
invalid_ciks = []
for cik in ciks_input.split(','):
    cik_stripped = cik.strip()
    if cik_stripped.isdigit():
        ciks.append(cik_stripped)
    elif cik_stripped: # Avoid adding empty strings
        invalid_ciks.append(cik_stripped)

if invalid_ciks:
     st.sidebar.warning(f"Ignoring invalid CIKs: {', '.join(invalid_ciks)}")


# Select Quarter
current_year = datetime.now().year
years = list(range(current_year, current_year - 10, -1)) # Last 10 years
quarters = [1, 2, 3, 4]

# Default to Q4 of previous year if current date is before first filing deadline (approx Feb 15)
# Otherwise default to most recently *completed* quarter for which filings *should* be available (allowing for 45d lag)
today = datetime.now().date()
month = today.month
default_q = 4
default_year = current_year - 1
if month >= 2 and month < 5: # Data for Q4 prev year filed by Feb 15
     default_q = 4
     default_year = current_year - 1
if month >= 5 and month < 8: # Data for Q1 current year filed by May 15
     default_q = 1
     default_year = current_year
if month >= 8 and month < 11: # Data for Q2 current year filed by Aug 14
     default_q = 2
     default_year = current_year
if month >= 11: # Data for Q3 current year filed by Nov 14
     default_q = 3
     default_year = current_year

# Set default year index, handling case where default_year might not be in list (unlikely here)
try:
    default_year_index = years.index(default_year)
except ValueError:
     default_year_index = 0 # Default to most recent year if calc fails

selected_year = st.sidebar.selectbox("Year", years, index=default_year_index)
# Set default quarter index (adjusting for 0-based index)
selected_quarter = st.sidebar.selectbox("Quarter", quarters, index=default_q - 1)

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
    * Data is lagged (filed up to 45 days after quarter end).
    * Only includes 'Section 13(f)' securities (mostly US-listed stocks/ETFs/options).
    * Excludes mutual funds, most bonds, shorts, non-US listed international stocks.
    * CIKs must be entered manually.
    * Stock view searches only within the managers specified above.
    * Data accuracy depends on filer submissions. Values in $USD.
    * XML parsing may fail for complex/non-standard filings.
    """
)

# --- Main Application Logic ---

# Use Streamlit's session state for caching fetched data to avoid re-fetching when inputs change slightly
if 'holdings_cache' not in st.session_state:
    st.session_state.holdings_cache = {}

def load_manager_data(cik_list, year, quarter):
    """Loads current and previous quarter data for multiple managers using Streamlit functions."""
    prev_year, prev_quarter = get_previous_quarter(year, quarter)
    loaded_data = {} # {cik: {'current': df, 'previous': df}}
    progress_bar = st.progress(0, text="Loading manager data...")
    total_ciks = len(cik_list)

    for i, cik in enumerate(cik_list):
        progress_text = f"Processing CIK: {cik} ({i+1}/{total_ciks})"
        progress_bar.progress((i + 1) / total_ciks, text=progress_text)

        st.write(f"--- Loading data for Manager CIK: {cik} ---")

        # Fetch current quarter data using the cached function
        df_curr = get_holdings_data(cik, year, quarter)
        st.write(f"Q{quarter} {year} holdings loaded: {len(df_curr)} records found.")

        # Fetch previous quarter data using the cached function
        df_prev = get_holdings_data(cik, prev_year, prev_quarter)
        st.write(f"Q{prev_quarter} {prev_year} holdings loaded: {len(df_prev)} records found.")

        loaded_data[cik] = {'current': df_curr, 'previous': df_prev}

    progress_bar.empty() # Remove progress bar when done
    return loaded_data

# --- Display Logic ---
if not ciks:
    st.warning("Please enter at least one valid CIK in the sidebar.")
else:
    # Load data for all selected CIKs first
    # Use a spinner while loading all data
    with st.spinner(f"Loading data for CIKs: {', '.join(ciks)}... Please wait."):
         all_manager_data = load_manager_data(ciks, selected_year, selected_quarter)

    if selected_view == "Manager View":
        # FIX: Use selected_quarter
        st.header(f"Manager View - {selected_year} Q{selected_quarter}")
        # Allow selecting which loaded CIK to view
        if ciks:
            selected_cik = st.selectbox("Select Manager CIK to Display", ciks)
        else:
            selected_cik = None # Should not happen if check above works

        if selected_cik and selected_cik in all_manager_data:
            data = all_manager_data[selected_cik]
            df_curr = data['current']
            df_prev = data['previous']

            if df_curr is None: # Check if loading failed for current
                st.error(f"Could not load current quarter data for CIK {selected_cik}. See warnings above.")
            elif not df_curr.empty:
                st.subheader(f"Holdings Changes for CIK: {selected_cik} (Q{selected_quarter} {selected_year} vs Previous Q{get_previous_quarter(selected_year, selected_quarter)[1]} {get_previous_quarter(selected_year, selected_quarter)[0]})")

                # Check if previous data is available
                if df_prev is None:
                    st.warning(f"Could not load previous quarter data for CIK {selected_cik}. Changes cannot be calculated.")
                    df_changes = pd.DataFrame() # Or display current holdings only
                else:
                     df_changes = calculate_holding_changes(df_curr, df_prev)

                if not df_changes.empty:
                     # Display summary of changes
                     st.metric("Total Holdings Reported (Current)", len(df_curr) if df_curr is not None else 0) # Use df_curr length
                     # Handle case where df_changes might be empty even if df_curr is not
                     if 'change_type' in df_changes.columns:
                          change_counts = df_changes['change_type'].value_counts()
                     else:
                          change_counts = pd.Series(dtype=int)

                     col1, col2, col3, col4, col5 = st.columns(5)
                     col1.metric("New", change_counts.get('New', 0))
                     col2.metric("Increased", change_counts.get('Increased', 0))
                     col3.metric("Decreased", change_counts.get('Decreased', 0))
                     col4.metric("Exited", change_counts.get('Exited', 0))
                     col5.metric("Unchanged/Value", change_counts.get('Unchanged', 0) + change_counts.get('Value Change Only', 0))


                     # Format columns for display
                     df_display = df_changes[[
                         'cusip', 'nameOfIssuer', 'change_type',
                         'sshPrnamt', 'value',
                         'change_shares', 'change_value', 'change_pct'
                     ]].sort_values(by=['change_type', 'value'], ascending=[True, False])

                     # Improve formatting for display
                     st.dataframe(df_display.style.format({
                         'value': "${:,.0f}",
                         'change_shares': "{:,.0f}",
                         'change_value': "${:,.0f}",
                         'change_pct': "{:.1f}%"
                     }).format(precision=1, na_rep='N/A', subset=['change_pct']), # Handle potential NAs in percentage
                     use_container_width=True)

                elif df_curr is not None and not df_curr.empty and df_prev is not None and df_prev.empty:
                      # Only current data available, previous was empty (or first filing)
                      st.info(f"Previous quarter data not found or empty for CIK {selected_cik}. Displaying current holdings as 'New'.")
                      df_curr['change_type'] = 'New' # Mark all as new
                      st.dataframe(df_curr[['cusip', 'nameOfIssuer', 'change_type', 'sshPrnamt', 'value']].style.format({'value': "${:,.0f}"}), use_container_width=True)

                elif df_curr is not None and not df_curr.empty:
                     # No changes calculated, but current data exists (and previous exists but maybe identical)
                     st.info(f"No significant changes detected or previous data unavailable/identical for CIK {selected_cik}. Displaying current holdings.")
                     st.dataframe(df_curr[['cusip', 'nameOfIssuer', 'sshPrnamt', 'value']].style.format({'value': "${:,.0f}"}), use_container_width=True)

            # Handle case where current data is explicitly empty after successful fetch/parse
            elif df_curr is not None and df_curr.empty:
                 st.info(f"No holdings data was found in the filing for CIK {selected_cik} for the selected quarter ({selected_year} Q{selected_quarter}). The manager might have held no reportable assets or filed an NT (Notice) report.")
            # else: handled by df_curr is None check above


    elif selected_view == "Stock View":
        # FIX: Use selected_quarter
        st.header(f"Stock View - {selected_year} Q{selected_quarter}")

        if not stock_cusip_input or len(stock_cusip_input) != 9:
            st.warning("Please enter a valid 9-character CUSIP in the sidebar for the Stock View.")
        elif not all_manager_data:
             st.warning("No manager data loaded. Please check CIKs and selected quarter.")
        else:
            st.subheader(f"Activity in Stock CUSIP: {stock_cusip_input} by Selected Managers")
            stock_activity = []
            stock_name = "Unknown Stock" # Placeholder

            for cik, data in all_manager_data.items():
                 df_curr = data['current']
                 df_prev = data['previous']

                 # Skip if current data failed to load
                 if df_curr is None:
                      logging.warning(f"Skipping CIK {cik} for stock view due to missing current data.")
                      continue

                 # Calculate changes specific to this manager for context
                 # Need previous data to calculate changes
                 if df_prev is None:
                     logging.warning(f"Skipping change calculation for CIK {cik} in stock view due to missing previous data.")
                     # Check if stock exists only in current data (New position)
                     current_stock_row = df_curr[df_curr['cusip'] == stock_cusip_input]
                     if not current_stock_row.empty:
                          activity = current_stock_row.iloc[0].to_dict()
                          activity['cik'] = cik
                          activity['change_type'] = 'New (Prev data N/A)'
                          activity['change_shares'] = activity['sshPrnamt']
                          activity['change_value'] = activity['value']
                          activity['change_pct'] = 100.0
                          stock_activity.append(activity)
                          if stock_name == "Unknown Stock" and pd.notna(activity['nameOfIssuer']):
                              stock_name = activity['nameOfIssuer']
                     continue # Move to next CIK


                 df_changes = calculate_holding_changes(df_curr, df_prev)

                 # Find the specific stock in the changes df
                 stock_row = df_changes[df_changes['cusip'] == stock_cusip_input]

                 if not stock_row.empty:
                     activity = stock_row.iloc[0].to_dict()
                     activity['cik'] = cik
                     stock_activity.append(activity)
                     # Try to get a consistent stock name
                     if stock_name == "Unknown Stock" and pd.notna(activity['nameOfIssuer']):
                         stock_name = activity['nameOfIssuer']
                 # Note: calculate_holding_changes should handle exited positions already

            if stock_activity:
                st.subheader(f"Activity for {stock_name} (CUSIP: {stock_cusip_input})")
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
                }).format(precision=1, na_rep='N/A', subset=['change_pct']), # Handle potential NAs in percentage
                use_container_width=True)
            else:
                st.info(f"No activity found for CUSIP {stock_cusip_input} among the selected managers ({', '.join(ciks)}) for the period {selected_year} Q{selected_quarter} vs Previous.")
