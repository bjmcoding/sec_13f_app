# requirements.txt:
# streamlit
# requests
# pandas
# lxml
# beautifulsoup4
# cachetools # Optional, as Streamlit caching is used

import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
from lxml import etree # Use lxml for robust parsing
from bs4 import BeautifulSoup
import re
import time
# Removed: from ratelimiter import RateLimiter # No longer needed/used
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
    # Class variables indented one level
    BASE_URL = "https://www.sec.gov"
    ARCHIVES_URL = f"{BASE_URL}/Archives/edgar/data"
    # Corrected API endpoint using data.sec.gov
    SUBMISSIONS_API_URL = f"https://data.sec.gov/submissions/CIK{{cik}}.json"

    # Method definition indented one level
    def __init__(self, user_agent): # <<< Ensure this line has correct indentation (e.g., 4 spaces)
        # Code inside method indented two levels
        self.user_agent = user_agent
        self.session = self._create_session()

    # Method definition indented one level
    def _create_session(self):
        """Creates a requests session with retry logic."""
        session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=1, # E.g., 1s, 2s, 4s, 8s, 16s
            status_forcelist=[429, 500, 502, 503, 504], # Retry on these codes
            allowed_methods=["HEAD", "GET", "OPTIONS"] # Use allowed_methods for newer urllib3
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        session.headers.update({"User-Agent": self.user_agent})
        return session

    # Method definition indented one level
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
            # Display error in Streamlit context if possible, otherwise just log
            try:
                st.error(f"Error fetching data from SEC EDGAR: {url}. Reason: {e}. Check CIK/Accession# and SEC status.")
            except Exception: # Handle cases where st is not available (e.g., background thread)
                 pass
            return None

    # Method definition indented one level
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

    # Method definition indented one level
    def find_latest_13f_hr_accession(self, cik, report_year, report_quarter):
        """
        Finds the accession number for the latest 13F-HR filing for a specific
        CIK and reporting period (year/quarter).
        """
        submissions_data = self.get_submissions(cik)
        if not submissions_data or 'filings' not in submissions_data or 'recent' not in submissions_data['filings']:
            st.warning(f"No recent filings found or submission data incomplete for CIK {cik}.")
            return None

        target_report_date_str = self._get_quarter_end_date(report_year, report_quarter)
        if not target_report_date_str:
            return None

        # Look through recent filings for a 13F-HR matching the report date
        filings = submissions_data['filings']['recent']
        latest_matching_filing = None
        latest_filing_date = datetime.min

        # Check if keys exist before accessing - robustness
        if 'accessionNumber' not in filings or \
           'form' not in filings or \
           'reportDate' not in filings or \
           'filingDate' not in filings:
            st.error(f"Unexpected structure in submissions data for CIK {cik}. Missing required keys.")
            logging.error(f"Unexpected structure in submissions data for CIK {cik}. Recent filings keys: {filings.keys()}")
            return None

        num_filings = len(filings['accessionNumber'])
        for i in range(num_filings):
            # Basic check if all lists have the same length
            if not (len(filings['form']) == num_filings and \
                    len(filings['reportDate']) == num_filings and \
                    len(filings['filingDate']) == num_filings):
                 st.error(f"Inconsistent lengths in filings data for CIK {cik}. Skipping.")
                 logging.error(f"Inconsistent lengths in filings data for CIK {cik}")
                 return None

            form = filings['form'][i]
            # We want 13F-HR (Holdings Report) or 13F-HR/A (Amendment)
            if form in ["13F-HR", "13F-HR/A"]:
                report_date = filings['reportDate'][i]
                filing_date_str = filings['filingDate'][i]

                if report_date == target_report_date_str:
                    try:
                        current_filing_date = datetime.strptime(filing_date_str, '%Y-%m-%d')
                        # Get the *latest filed* report for that quarter end
                        if latest_matching_filing is None or current_filing_date >= latest_filing_date: # Use >= to take latest amendment if filed same day
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

    # Method definition indented one level
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

    # Method definition indented one level
    def _get_primary_doc_url(self, cik, accession_number):
        """Constructs the URL for the primary submission document (.txt)."""
        cik_padded = str(cik).zfill(10)
        accession_no_dash = accession_number.replace('-', '')
        return f"{self.ARCHIVES_URL}/{cik_padded}/{accession_no_dash}/{accession_number}.txt"

    # Method definition indented one level
    def _get_html_index_url(self, cik, accession_number):
        """Constructs the URL for the HTML index page."""
        cik_padded = str(cik).zfill(10)
        accession_no_dash = accession_number.replace('-', '')
        return f"{self.ARCHIVES_URL}/{cik_padded}/{accession_no_dash}/{accession_number}-index.htm" # Note: sometimes .html

    # Method definition indented one level
    def _fetch_primary_doc(self, cik, accession_number):
        """Fetches the content of the primary submission document."""
        url = self._get_primary_doc_url(cik, accession_number)
        response = self._make_request(url)
        return response.text if response else None

    # Method definition indented one level
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
            logging.info(f"Retrying HTML index with .html suffix: {url_html}")
            response_html = self._make_request(url_html)
            return response_html.text if response_html else None

    # Method definition indented one level
    # Use @st.cache_data for Streamlit's built-in caching mechanism
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
            # Error should have been displayed by _fetch_primary_doc
            return None

        # --- Step 3: Parse Primary Document Manifest ---
        try:
            # Regex to find <DOCUMENT> blocks, tolerant to variations
            document_blocks = re.findall(r'<DOCUMENT>(.*?)</DOCUMENT>', primary_doc_content, re.DOTALL | re.IGNORECASE)
            logging.info(f"Found {len(document_blocks)} <DOCUMENT> blocks in primary doc for {accession_number}.")

            for block in document_blocks:
                # More robust regex to handle potential variations in whitespace/newlines
                type_match = re.search(r'<TYPE>\s*(.*?)\s*($|</SEC-HEADER>)', block, re.IGNORECASE | re.MULTILINE)
                filename_match = re.search(r'<FILENAME>\s*(.*?)\s*\n', block, re.IGNORECASE)

                if type_match and filename_match:
                    doc_type = type_match.group(1).strip().upper()
                    doc_filename = filename_match.group(1).strip()

                    # Check if the type is INFORMATION TABLE (or common variations)
                    if 'INFORMATION TABLE' in doc_type or 'FORM 13F INFORMATION TABLE' in doc_type or doc_type == 'XML':
                         # Ensure it's likely an XML file
                        if doc_filename.lower().endswith('.xml'):
                            logging.info(f"Found Info Table via Manifest: Type='{doc_type}', Filename='{doc_filename}' for {accession_number}")
                            return doc_filename
                        else:
                             logging.warning(f"Found INFORMATION TABLE type, but filename '{doc_filename}' is not XML for {accession_number}. Skipping.")
                    # Handle cases where Type might just be XML and description gives context
                    elif doc_type == 'XML':
                         desc_match = re.search(r'<DESCRIPTION>\s*(.*?)\s*\n', block, re.IGNORECASE)
                         if desc_match:
                              doc_desc = desc_match.group(1).strip().upper()
                              if 'INFORMATION TABLE' in doc_desc:
                                   if doc_filename.lower().endswith('.xml'):
                                       logging.info(f"Found Info Table via Manifest (XML type, Description check): Desc='{doc_desc}', Filename='{doc_filename}' for {accession_number}")
                                       return doc_filename


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
                # Try finding by specific class often used by SEC
                doc_table = soup.find('table', class_='tableFile')
                if not doc_table:
                    # Fallback to summary attribute check
                    for table in soup.find_all('table'):
                        summary = table.get('summary', '').lower()
                        if 'document format files' in summary or 'submission documents' in summary:
                            doc_table = table
                            break
                # Fallback: check table headers if other methods fail
                if not doc_table:
                     for table in soup.find_all('table'):
                         headers = [th.get_text(strip=True).lower() for th in table.find_all('th')]
                         if 'document' in headers and ('type' in headers or 'description' in headers):
                              doc_table = table
                              break # Take the first plausible table

                if doc_table:
                    logging.info(f"Found potential document table in HTML index for {accession_number}.")
                    rows = doc_table.find_all('tr')
                    for row in rows[1:]: # Skip header row
                        cells = row.find_all('td')
                        if len(cells) >= 3: # Need at least Seq, Document, Type/Description
                            try:
                                doc_link_cell = cells[1] # Usually the second cell has the link
                                type_cell = cells[3] if len(cells) > 3 else cells[2] # Type/Description often 3rd or 4th
                                description_cell = cells[2] if len(cells) > 3 else type_cell # Try description too

                                type_text = type_cell.get_text(strip=True).upper()
                                desc_text = description_cell.get_text(strip=True).upper()
                                doc_link = doc_link_cell.find('a')

                                if doc_link and doc_link.has_attr('href'):
                                    filename = doc_link.get_text(strip=True)
                                    # Check if type or description indicates information table
                                    if 'INFORMATION TABLE' in type_text or '13F TABLE' in type_text or \
                                       'INFORMATION TABLE' in desc_text or '13F TABLE' in desc_text:
                                         if filename.lower().endswith('.xml'):
                                            logging.info(f"Found Info Table via HTML Index: Type='{type_text}', Desc='{desc_text}', Filename='{filename}' for {accession_number}")
                                            return filename
                                         else:
                                            logging.warning(f"Found INFORMATION TABLE type/desc in HTML index, but filename '{filename}' is not XML for {accession_number}. Skipping.")
                            except IndexError:
                                logging.warning(f"Skipping row in HTML index table due to unexpected cell count for {accession_number}")
                                continue # Skip malformed rows
            except Exception as e:
                logging.error(f"Error parsing HTML index page for {accession_number}: {e}")
                # Continue to next fallback

        # --- Step 5: Fallback - Check Common Default Filenames (Least Reliable) ---
        logging.warning(f"HTML index parsing failed or yielded no XML. Trying default filenames for {accession_number}.")
        cik_padded = str(cik).zfill(10)
        accession_no_dash = accession_number.replace('-', '')
        base_path = f"{self.ARCHIVES_URL}/{cik_padded}/{accession_no_dash}"
        default_filenames = ['infotable.xml', 'form13fInfoTable.xml', 'information_table.xml', 'form13finformationtable.xml'] # Add others if known

        for filename in default_filenames:
            url = f"{base_path}/{filename}"
            time.sleep(SEC_REQUEST_DELAY) # Still need rate limiting
            try:
                # Use HEAD request to check existence without downloading fully
                response = self.session.head(url, timeout=5)
                if response.status_code == 200:
                     # Double check content type if possible
                     content_type = response.headers.get('Content-Type', '').lower()
                     if 'xml' in content_type or 'text' in content_type: # Accept text/plain too as sometimes servers mislabel XML
                        logging.info(f"Found Info Table via Default Filename Check: '{filename}' for {accession_number}")
                        return filename
                     else:
                         logging.warning(f"HEAD request for default '{filename}' returned 200 but unexpected Content-Type: {content_type} for {accession_number}")
            except requests.exceptions.RequestException as e:
                # Log minor errors, as failure here is expected sometimes
                logging.debug(f"HEAD request failed for default {filename} for {accession_number}: {e}")
                continue # Try next default filename

        logging.error(f"Could not identify Information Table XML filename for {accession_number} using any method.")
        st.error(f"Failed to find the Information Table XML file within filing {accession_number}. The filing might be structured unusually, be very old, or missing the XML table.")
        return None

    # Method definition indented one level
    def _get_info_table_xml_url(self, cik, accession_number, xml_filename):
        """Constructs the full URL to the identified Information Table XML."""
        cik_padded = str(cik).zfill(10)
        accession_no_dash = accession_number.replace('-', '')
        # Ensure filename doesn't have leading slashes if parsed incorrectly
        xml_filename_cleaned = xml_filename.lstrip('/')
        return f"{self.ARCHIVES_URL}/{cik_padded}/{accession_no_dash}/{xml_filename_cleaned}"

    # Method definition indented one level
    def _fetch_info_table_xml(self, xml_url):
        """Fetches the content of the Information Table XML."""
        response = self._make_request(xml_url)
        # Use response.content for potentially binary XML data
        return response.content if response else None

    # Method definition indented one level
    def parse_info_table_xml(self, xml_content):
        """Parses the 13F Information Table XML into a pandas DataFrame."""
        if not xml_content:
            return pd.DataFrame() # Return empty DataFrame if no content

        try:
            # Use BytesIO to handle the XML content (which might be bytes)
            # Attempt to detect encoding, default to utf-8
            try:
                decoded_content = xml_content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    decoded_content = xml_content.decode('iso-8859-1') # Try latin-1 as fallback
                except UnicodeDecodeError:
                    logging.error("Failed to decode XML content with utf-8 or iso-8859-1.")
                    st.error("Failed to decode XML content. Filing may use an unexpected encoding.")
                    return pd.DataFrame()

            # Remove potential Byte Order Mark (BOM) which can interfere with lxml
            if decoded_content.startswith('\ufeff'):
                decoded_content = decoded_content[1:]

            # Parse from the decoded string
            parser = etree.XMLParser(recover=True) # Use recover=True to handle minor errors
            tree = etree.fromstring(decoded_content.encode('utf-8'), parser=parser) # Re-encode to bytes for fromstring

            # Define XML namespaces (often present in 13F XML)
            # These might vary slightly, common ones are included
            # Extract namespaces dynamically if possible, otherwise use common defaults
            ns = tree.nsmap
            # Ensure a default namespace key exists if needed, mapping to the main URI
            if None in ns:
                ns['defns'] = ns.pop(None) # Use 'defns' as placeholder if default ns exists
            else:
                 # Check for common SEC namespace URIs if nsmap is empty or doesn't contain them
                 common_ns_uri = 'http://www.sec.gov/edgar/document/thirteenf/informationtable'
                 if not any(uri == common_ns_uri for uri in ns.values()):
                     ns['ns'] = common_ns_uri # Add a default 'ns' prefix if missing


            # Helper to find elements with namespace handling
            def find_all_with_ns(element, path):
                 # Try with namespaces first
                try:
                     found = element.xpath(f".//{path}", namespaces=ns)
                     if found: return found
                except etree.XPathEvalError as e:
                     # Handle potential issues if path itself uses prefixes not in ns map
                     logging.warning(f"XPathEvalError for path '{path}': {e}. Trying without explicit ns.")
                     # Fallback below will handle prefix-less path

                 # If not found or error, try assuming path doesn't need prefixes (or uses default ns implicitly)
                 # This handles cases where nsmap might be incomplete or XML doesn't use prefixes
                 try:
                      # Remove prefixes if they exist in the path, assuming default ns or no ns
                      path_no_prefix = re.sub(r'\b\w+:', '', path)
                      found = element.xpath(f".//{path_no_prefix}")
                      if found: return found
                 except etree.XPathEvalError as e:
                      logging.error(f"XPathEvalError even after removing prefixes for path '{path}': {e}")

                 return [] # Return empty list if nothing found


            # Extract data for each holding ('infoTable' element)
            holdings = []
            # Find the infoTable elements - they might be prefixed or not
            info_tables = find_all_with_ns(tree, 'infoTable')
            if not info_tables:
                 # Fallback if namespace caused issues or element name is different
                 info_tables = tree.xpath("//*[local-name()='infoTable']")
                 if not info_tables:
                      logging.error("Could not find 'infoTable' elements in the XML.")
                      st.error("Could not find holdings ('infoTable' elements) in the parsed XML file.")
                      return pd.DataFrame()


            logging.info(f"Found {len(info_tables)} infoTable elements.")
            for table in info_tables:
                holding = {}
                # Helper to extract text safely
                def get_text(element_list):
                    return element_list[0].text.strip() if element_list and element_list[0].text is not None else None

                # Extract fields - use helper and handle missing elements gracefully
                holding['nameOfIssuer'] = get_text(find_all_with_ns(table, 'nameOfIssuer'))
                holding['titleOfClass'] = get_text(find_all_with_ns(table, 'titleOfClass'))
                holding['cusip'] = get_text(find_all_with_ns(table, 'cusip'))
                # Value is reported in thousands, multiply by 1000
                value_text = get_text(find_all_with_ns(table, 'value'))
                holding['value'] = int(value_text.replace(',', '')) * 1000 if value_text else 0
                # Shares or Principal Amount Data
                ssh_prnamt_elem = find_all_with_ns(table, 'shrsOrPrnAmt')
                if ssh_prnamt_elem:
                     holding['sshPrnamt'] = int(get_text(find_all_with_ns(ssh_prnamt_elem[0], 'sshPrnamt')).replace(',', '')) if get_text(find_all_with_ns(ssh_prnamt_elem[0], 'sshPrnamt')) else 0
                     holding['sshPrnamtType'] = get_text(find_all_with_ns(ssh_prnamt_elem[0], 'sshPrnamtType'))
                else: # Try older structure without sub-elements (less common now)
                     holding['sshPrnamt'] = int(get_text(find_all_with_ns(table, 'sshPrnamt')).replace(',', '')) if get_text(find_all_with_ns(table, 'sshPrnamt')) else 0
                     holding['sshPrnamtType'] = get_text(find_all_with_ns(table, 'sshPrnamtType'))

                holding['investmentDiscretion'] = get_text(find_all_with_ns(table, 'investmentDiscretion'))
                # Voting Authority
                voting_auth_elem = find_all_with_ns(table, 'votingAuthority')
                if voting_auth_elem:
                    holding['votingAuthSole'] = int(get_text(find_all_with_ns(voting_auth_elem[0], 'Sole')).replace(',', '')) if get_text(find_all_with_ns(voting_auth_elem[0], 'Sole')) else 0
                    holding['votingAuthShared'] = int(get_text(find_all_with_ns(voting_auth_elem[0], 'Shared')).replace(',', '')) if get_text(find_all_with_ns(voting_auth_elem[0], 'Shared')) else 0
                    holding['votingAuthNone'] = int(get_text(find_all_with_ns(voting_auth_elem[0], 'None')).replace(',', '')) if get_text(find_all_with_ns(voting_auth_elem[0], 'None')) else 0
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

            # Filter out rows with missing CUSIP as they cannot be reliably tracked
            original_len = len(df)
            df = df.dropna(subset=['cusip'])
            df = df[df['cusip'] != ''] # Also remove empty strings
            if len(df) < original_len:
                logging.warning(f"Removed {original_len - len(df)} holdings with missing CUSIP.")


            logging.info(f"Successfully parsed {len(df)} holdings with valid CUSIP from XML.")
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

# Function definition indented at root level
@st.cache_data(ttl=3600) # Cache results for 1 hour
def get_holdings_data(cik, year, quarter):
    """
    Orchestrates fetching and parsing 13F holdings data for a specific CIK and quarter.
    """
    # Ensure CIK is a string for consistency
    cik_str = str(cik).strip()
    if not cik_str.isdigit():
        st.error(f"Invalid CIK format provided: {cik}. Should be numeric.")
        return pd.DataFrame()

    handler = SECEdgarHandler(SEC_USER_AGENT)
    accession_number = handler.find_latest_13f_hr_accession(cik_str, year, quarter)
    if not accession_number:
        # Warning/error should be shown by find_latest_13f_hr_accession
        return pd.DataFrame()

    xml_filename = handler.find_info_table_xml_filename(cik_str, accession_number)
    if not xml_filename:
         # Error shown by find_info_table_xml_filename
         return pd.DataFrame()

    xml_url = handler._get_info_table_xml_url(cik_str, accession_number, xml_filename)
    xml_content = handler._fetch_info_table_xml(xml_url)
    if not xml_content:
         st.error(f"Failed to download the XML file: {xml_url}")
         return pd.DataFrame()

    return handler.parse_info_table_xml(xml_content)

# Function definition indented at root level
def get_previous_quarter(year, quarter):
    """ Calculates the year and quarter for the previous period. """
    if quarter == 1:
        return year - 1, 4
    else:
        return year, quarter - 1

# Function definition indented at root level
# Use @st.cache_data for Streamlit's built-in caching mechanism
@st.cache_data(ttl=3600)
def calculate_holding_changes(_df_curr, _df_prev):
    """
    Compares holdings between two quarters (DataFrames) and calculates changes.
    Uses _df_curr, _df_prev because Streamlit caching decorators modify argument handling.
    Handles potential empty DataFrames more robustly.
    """
    # Ensure DataFrames are not None and handle empty cases
    if _df_curr is None: _df_curr = pd.DataFrame()
    if _df_prev is None: _df_prev = pd.DataFrame()

    # Check if essential columns exist
    required_cols = ['cusip', 'sshPrnamt', 'value', 'nameOfIssuer']
    if not all(col in _df_curr.columns for col in required_cols) and not _df_curr.empty:
        logging.error(f"Current holdings DataFrame missing required columns. Found: {_df_curr.columns}")
        st.error("Current holdings data is missing expected columns (e.g., cusip, sshPrnamt, value). Cannot calculate changes.")
        # Return only current data if possible
        return _df_curr[['cusip', 'nameOfIssuer', 'sshPrnamt', 'value']].assign(
            change_type='Error', change_shares=0, change_value=0, change_pct=pd.NA
        ) if all(col in _df_curr.columns for col in ['cusip', 'nameOfIssuer', 'sshPrnamt', 'value']) else pd.DataFrame()


    if not all(col in _df_prev.columns for col in required_cols) and not _df_prev.empty:
        logging.warning(f"Previous holdings DataFrame missing required columns. Found: {_df_prev.columns}. Treating as empty for change calculation.")
        _df_prev = pd.DataFrame() # Treat as empty if structure is wrong

    if _df_curr.empty and _df_prev.empty:
         return pd.DataFrame() # Nothing to compare

    # If only current data exists
    if not _df_curr.empty and _df_prev.empty:
        df_out = _df_curr[['cusip', 'nameOfIssuer', 'sshPrnamt', 'value']].copy()
        df_out['change_type'] = 'New/Unknown' # Mark as New or Unknown if prev data missing
        df_out['change_shares'] = df_out['sshPrnamt']
        df_out['change_value'] = df_out['value']
        df_out['change_pct'] = 100.0
        return df_out

    # If only previous data exists
    if _df_curr.empty and not _df_prev.empty:
        df_out = _df_prev[['cusip', 'nameOfIssuer', 'sshPrnamt', 'value']].copy()
        df_out['change_type'] = 'Exited/Unknown' # Mark as Exited or Unknown if curr data missing
        df_out['change_shares'] = -df_out['sshPrnamt']
        df_out['change_value'] = -df_out['value']
        df_out['change_pct'] = -100.0
        # Set current shares/value to 0 for clarity
        df_out['sshPrnamt'] = 0
        df_out['value'] = 0
        return df_out


    # Merge based on CUSIP - ensure CUSIP columns exist
    if 'cusip' not in _df_curr.columns or 'cusip' not in _df_prev.columns:
         st.error("CUSIP column missing from one or both DataFrames. Cannot calculate changes.")
         return pd.DataFrame()

    # Ensure required columns exist before merge selection
    cols_curr = [col for col in ['cusip', 'nameOfIssuer', 'sshPrnamt', 'value'] if col in _df_curr.columns]
    cols_prev = [col for col in ['cusip', 'nameOfIssuer', 'sshPrnamt', 'value'] if col in _df_prev.columns]

    df_merged = pd.merge(
        _df_curr[cols_curr],
        _df_prev[cols_prev],
        on='cusip',
        how='outer',
        suffixes=('_curr', '_prev')
    )

    changes = []
    for _, row in df_merged.iterrows():
        cusip = row['cusip']
        # Try to get name from current, fallback to previous
        name = row.get('nameOfIssuer_curr', row.get('nameOfIssuer_prev', 'N/A'))

        shares_curr = row.get('sshPrnamt_curr', 0) if pd.notna(row.get('sshPrnamt_curr')) else 0
        value_curr = row.get('value_curr', 0) if pd.notna(row.get('value_curr')) else 0
        shares_prev = row.get('sshPrnamt_prev', 0) if pd.notna(row.get('sshPrnamt_prev')) else 0
        value_prev = row.get('value_prev', 0) if pd.notna(row.get('value_prev')) else 0

        change_type = 'N/A'
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
            if shares_prev != 0: # Avoid division by zero
                 change_pct = (shares_curr - shares_prev) / shares_prev * 100.0
            else: # Should not happen if shares_prev > 0, but safety check
                 change_pct = 100.0 if shares_curr > 0 else 0.0

            if abs(change_shares) > 0: # Only mark changed if shares actually changed
                 change_type = 'Increased' if change_shares > 0 else 'Decreased'
            else:
                 # Optional: Check for significant value change even if shares are same (e.g., due to splits/adjustments not reflected)
                 if abs(value_curr - value_prev) > (value_prev * 0.01) : # e.g. > 1% value change
                     change_type = 'Value Changed'
                 else:
                     change_type = 'Unchanged'
        elif shares_curr == 0 and shares_prev == 0:
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

    if not changes:
         return pd.DataFrame() # Return empty if no changes calculated

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
# Validate CIKs are digits before processing
ciks = [cik.strip() for cik in ciks_input.split(',') if cik.strip().isdigit()]

# Select Quarter
current_dt = datetime.now()
current_year = current_dt.year
years = list(range(current_year, current_year - 10, -1)) # Last 10 years
quarters = [1, 2, 3, 4]

selected_year = st.sidebar.selectbox("Year", years)

# Default to the most recently fully completed quarter for which filings *might* be available
# (assuming ~45 day lag + buffer)
default_q = 4
default_year = selected_year - 1
if current_dt.month >= 11 or (current_dt.month == 10 and current_dt.day >= 15): # Q3 likely done
    default_q = 3
    default_year = selected_year
elif current_dt.month >= 8 or (current_dt.month == 7 and current_dt.day >= 15): # Q2 likely done
    default_q = 2
    default_year = selected_year
elif current_dt.month >= 5 or (current_dt.month == 4 and current_dt.day >= 15): # Q1 likely done
    default_q = 1
    default_year = selected_year

# Adjust default year if current selection is different
if selected_year != default_year:
     # If user selects a different year, default to Q4 of that year initially
     default_q_index = 3 # Index for Q4
else:
     default_q_index = default_q - 1

selected_quarter = st.sidebar.selectbox("Quarter", quarters, index=default_q_index)


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
    * Data reflects holdings **as of** the selected quarter-end (e.g., March 31), filed up to 45 days later.
    * Only includes 'Section 13(f)' securities (mostly US-listed stocks/ETFs).
    * Excludes mutual funds, most bonds, shorts, international stocks (non-US listed), options details beyond underlying shares.
    * CIKs must be entered manually.
    * Stock view searches **only** within the managers listed above.
    * Data accuracy depends on filer submissions. Values in $USD.
    """
)

# --- Main Application Logic ---

# Data cache dictionary using Streamlit's session state
if 'holdings_cache' not in st.session_state:
    st.session_state.holdings_cache = {}

def load_manager_data(cik_list, year, quarter):
    """Loads current and previous quarter data for multiple managers into the cache."""
    prev_year, prev_quarter = get_previous_quarter(year, quarter)
    loaded_data = {} # {cik: {'current': df, 'previous': df}}
    at_least_one_success = False

    st.markdown(f"**Attempting to load data for {len(cik_list)} manager(s) for {year} Q{quarter} and {prev_year} Q{prev_quarter}**")

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, cik in enumerate(cik_list):
        cik_str = str(cik).strip() # Ensure string format
        status_text.text(f"Processing CIK: {cik_str} ({i+1}/{len(cik_list)})...")

        # --- Fetch Current Quarter ---
        cache_key_curr = f"{cik_str}_{year}_Q{quarter}"
        df_curr = None
        if cache_key_curr not in st.session_state.holdings_cache:
             logging.info(f"Cache miss for {cache_key_curr}. Fetching...")
             df_curr = get_holdings_data(cik_str, year, quarter) # Function handles @st.cache_data
             # Store even if empty to avoid re-fetching failed attempts within session
             st.session_state.holdings_cache[cache_key_curr] = df_curr.copy() if df_curr is not None else pd.DataFrame()
        else:
             logging.info(f"Cache hit for {cache_key_curr}.")
             df_curr = st.session_state.holdings_cache[cache_key_curr]
             # Display message if using cached empty result from previous failed attempt
             if df_curr.empty:
                 st.info(f"Using cached empty result for {cik_str} - {year} Q{quarter} (likely failed fetch previously).")
             else:
                 st.info(f"Using cached data for {cik_str} - {year} Q{quarter}")


        # --- Fetch Previous Quarter ---
        cache_key_prev = f"{cik_str}_{prev_year}_Q{prev_quarter}"
        df_prev = None
        if cache_key_prev not in st.session_state.holdings_cache:
             logging.info(f"Cache miss for {cache_key_prev}. Fetching...")
             df_prev = get_holdings_data(cik_str, prev_year, prev_quarter)
             # Store even if empty
             st.session_state.holdings_cache[cache_key_prev] = df_prev.copy() if df_prev is not None else pd.DataFrame()
        else:
             logging.info(f"Cache hit for {cache_key_prev}.")
             df_prev = st.session_state.holdings_cache[cache_key_prev]
             if df_prev.empty:
                  st.info(f"Using cached empty result for {cik_str} - {prev_year} Q{prev_quarter} (likely failed fetch previously).")
             else:
                  st.info(f"Using cached data for {cik_str} - {prev_year} Q{prev_quarter}")

        # Store results (even if empty)
        loaded_data[cik_str] = {'current': df_curr if df_curr is not None else pd.DataFrame(),
                                'previous': df_prev if df_prev is not None else pd.DataFrame()}
        if df_curr is not None and not df_curr.empty:
            at_least_one_success = True

        progress_bar.progress((i + 1) / len(cik_list))

    status_text.text(f"Finished loading data for {len(cik_list)} manager(s).")
    progress_bar.empty() # Remove progress bar after completion

    if not at_least_one_success and cik_list:
         st.warning("Could not successfully load current quarter data for any of the specified CIKs. Please verify CIKs, selected period, and SEC EDGAR status.")

    return loaded_data

# --- Display Logic ---
if not ciks:
    st.warning("Please enter at least one valid CIK (numeric) in the sidebar.")
else:
    # Load data for all selected CIKs first
    # Wrap in spinner for the overall loading process
    with st.spinner("Loading all manager data... This may take time depending on cache status and number of CIKs."):
        all_manager_data = load_manager_data(ciks, selected_year, selected_quarter)

    if selected_view == "Manager View":
        # Correct variable name used here
        st.header(f"Manager View - {selected_year} Q{selected_quarter}")
        # Ensure CIK list for selectbox uses the keys from loaded data (which are strings)
        available_ciks = list(all_manager_data.keys())
        if not available_ciks:
             st.warning("No data could be loaded for the selected CIKs.")
        else:
            selected_cik_str = st.selectbox("Select Manager CIK to Display", available_ciks)

            if selected_cik_str in all_manager_data:
                data = all_manager_data[selected_cik_str]
                df_curr = data['current']
                df_prev = data['previous']

                st.subheader(f"Analysis for CIK: {selected_cik_str}")

                if df_curr is not None and not df_curr.empty:
                    st.write(f"**Holdings Changes (Q{selected_quarter} {selected_year} vs Q{get_previous_quarter(selected_year, selected_quarter)[1]} {get_previous_quarter(selected_year, selected_quarter)[0]})**")
                    # Ensure calculate_holding_changes gets DataFrames, not None
                    df_changes = calculate_holding_changes(df_curr if df_curr is not None else pd.DataFrame(),
                                                        df_prev if df_prev is not None else pd.DataFrame())

                    if df_changes is not None and not df_changes.empty:
                         # Display summary of changes
                         st.metric("Total Holdings Reported (Current Q)", len(df_curr))
                         change_counts = df_changes['change_type'].value_counts()
                         col1, col2, col3, col4 = st.columns(4)
                         col1.metric("New Positions", change_counts.get('New', 0) + change_counts.get('New/Unknown', 0))
                         col2.metric("Increased Positions", change_counts.get('Increased', 0))
                         col3.metric("Decreased Positions", change_counts.get('Decreased', 0))
                         col4.metric("Exited Positions", change_counts.get('Exited', 0) + change_counts.get('Exited/Unknown', 0))


                         # Format columns for display
                         df_display_changes = df_changes[[
                             'cusip', 'nameOfIssuer', 'change_type',
                             'sshPrnamt', 'value', # Current values
                             'change_shares', 'change_value', 'change_pct'
                         ]].sort_values(by=['change_type', 'value'], ascending=[True, False]) # Sort for clarity

                         st.dataframe(df_display_changes.style.format({
                             'value': "${:,.0f}",
                             'sshPrnamt': "{:,.0f}",
                             'change_shares': "{:,.0f}",
                             'change_value': "${:,.0f}",
                             'change_pct': "{:.1f}%"
                         }), use_container_width=True)

                    else:
                         st.info(f"No changes calculated for CIK {selected_cik_str}. Displaying current holdings only.")
                         st.write(f"**Current Holdings (as of {selected_year}-Q{selected_quarter} end)**")
                         st.dataframe(df_curr[['cusip', 'nameOfIssuer', 'sshPrnamt', 'value']]
                                       .style.format({'value': "${:,.0f}", 'sshPrnamt': "{:,.0f}"}),
                                       use_container_width=True)

                elif df_curr is None:
                     st.error(f"There was an error loading current quarter data for CIK {selected_cik_str}.")
                else: # df_curr is an empty DataFrame
                    st.warning(f"No holdings data found or parsed for CIK {selected_cik_str} for the selected quarter ({selected_year} Q{selected_quarter}). The manager might not have filed or held reportable assets.")

            else:
                 # Should not happen if selectbox populated correctly
                 st.error("Selected CIK not found in loaded data.")


    elif selected_view == "Stock View":
        # Correct variable name used here
        st.header(f"Stock View - {selected_year} Q{selected_quarter}")

        if not stock_cusip_input or len(stock_cusip_input) != 9:
            st.warning("Please enter a valid 9-character CUSIP in the sidebar for the Stock View.")
        elif not all_manager_data:
             st.warning("No manager data loaded. Please check CIKs and selected quarter.")
        else:
            stock_cusip_input_upper = stock_cusip_input.upper() # Ensure comparison is case-insensitive
            st.subheader(f"Activity in Stock CUSIP: {stock_cusip_input_upper} by Selected Managers")
            stock_activity = []
            processed_ciks = 0

            for cik, data in all_manager_data.items():
                 processed_ciks += 1
                 logging.info(f"Processing Stock View for CIK {cik}, Stock {stock_cusip_input_upper}")
                 df_curr = data['current']
                 df_prev = data['previous']

                 # Ensure DataFrames are valid before proceeding
                 if df_curr is None and df_prev is None:
                      logging.warning(f"Both current and previous data are None for CIK {cik}. Skipping stock view.")
                      continue

                 # Calculate changes specific to this manager for context
                 df_changes = calculate_holding_changes(df_curr if df_curr is not None else pd.DataFrame(),
                                                     df_prev if df_prev is not None else pd.DataFrame())

                 if df_changes is not None and not df_changes.empty and 'cusip' in df_changes.columns:
                     # Find the specific stock in the changes df (case-insensitive comparison)
                     stock_row = df_changes[df_changes['cusip'].str.upper() == stock_cusip_input_upper]

                     if not stock_row.empty:
                         activity = stock_row.iloc[0].to_dict()
                         activity['cik'] = cik
                         stock_activity.append(activity)
                         logging.info(f"Found activity for {stock_cusip_input_upper} in CIK {cik} via changes df.")
                     else:
                          # Explicitly check if held only in previous (Exited case not caught if df_curr was None/empty)
                          if df_prev is not None and not df_prev.empty and 'cusip' in df_prev.columns:
                              prev_row = df_prev[df_prev['cusip'].str.upper() == stock_cusip_input_upper]
                              if not prev_row.empty:
                                   # Only add if not already captured by df_changes (which implies df_curr was empty)
                                   if df_curr is None or df_curr.empty or df_curr[df_curr['cusip'].str.upper() == stock_cusip_input_upper].empty:
                                        stock_activity.append({
                                            'cik': cik,
                                            'cusip': stock_cusip_input_upper,
                                            'nameOfIssuer': prev_row.iloc[0]['nameOfIssuer'],
                                            'sshPrnamt': 0, # Current shares are 0
                                            'value': 0,     # Current value is 0
                                            'change_type': 'Exited/Unknown',
                                            'change_shares': -prev_row.iloc[0]['sshPrnamt'],
                                            'change_value': -prev_row.iloc[0]['value'],
                                            'change_pct': -100.0
                                        })
                                        logging.info(f"Found exited activity for {stock_cusip_input_upper} in CIK {cik} via previous df check.")
                 elif df_curr is not None and not df_curr.empty and 'cusip' in df_curr.columns and (df_prev is None or df_prev.empty):
                      # Only current data exists, check if stock is present (New/Unknown case)
                      curr_row = df_curr[df_curr['cusip'].str.upper() == stock_cusip_input_upper]
                      if not curr_row.empty:
                           stock_activity.append({
                                'cik': cik,
                                'cusip': stock_cusip_input_upper,
                                'nameOfIssuer': curr_row.iloc[0]['nameOfIssuer'],
                                'sshPrnamt': curr_row.iloc[0]['sshPrnamt'],
                                'value': curr_row.iloc[0]['value'],
                                'change_type': 'New/Unknown',
                                'change_shares': curr_row.iloc[0]['sshPrnamt'],
                                'change_value': curr_row.iloc[0]['value'],
                                'change_pct': 100.0
                           })
                           logging.info(f"Found new/unknown activity for {stock_cusip_input_upper} in CIK {cik} via current df check.")


            if stock_activity:
                df_stock_view = pd.DataFrame(stock_activity)
                # Reorder and format columns
                df_stock_display = df_stock_view[[
                     'cik', 'cusip', 'nameOfIssuer', 'change_type',
                     'sshPrnamt', 'value', # Current values
                     'change_shares', 'change_value', 'change_pct'
                ]].sort_values(by=['change_type', 'value'], ascending=[True, False])

                st.dataframe(df_stock_display.style.format({
                     'value': "${:,.0f}",
                     'sshPrnamt': "{:,.0f}",
                     'change_shares': "{:,.0f}",
                     'change_value': "${:,.0f}",
                     'change_pct': "{:.1f}%"
                }), use_container_width=True)
            elif processed_ciks > 0: # Only show if we actually processed managers
                st.info(f"No activity found for CUSIP {stock_cusip_input_upper} among the selected managers ({', '.join(ciks)}) for the period Q{selected_quarter} {selected_year} vs Previous.")
            else:
                st.warning("Could not process any managers for stock view.")
