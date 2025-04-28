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
from io import StringIO, BytesIO
import logging
from datetime import datetime, timedelta

# --- Configuration ---
USER_AGENT_NAME = "BJMCODING"
USER_AGENT_EMAIL = "bmccormick.seo@gmail.com"
SEC_USER_AGENT = f"{USER_AGENT_NAME} {USER_AGENT_EMAIL}"
SEC_REQUEST_DELAY = 0.11 # seconds between requests

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Mapping Dictionaries (Simple Demo Version) ---
# NOTE: Expand these or replace with a dynamic lookup method for a real application
FUND_NAME_TO_CIK = {
    "Berkshire Hathaway Inc": "1067983", # Note: This CIK often files for subsidiaries, main holdings under 00001000180 might be needed too sometimes. Using the main one often listed.
    "Tiger Global Management LLC": "1167483", # Found via SEC search - check if correct CIK for 13F filings
    "Scion Asset Management, LLC": "1649339",
    "Pershing Square Capital Management, L.P.": "1336528",
    "Renaissance Technologies LLC": "1037389",
    "Bridgewater Associates, LP": "1066374",
    "Appaloosa LP": "1056188"
    # Add more fund names and their corresponding CIKs here
}

# Using Ticker -> CUSIP (primary identifier in 13F)
# NOTE: This is highly simplified. Tickers can change, map to multiple CUSIPs (classes), etc.
# CUSIPs sourced via web search - verify accuracy.
TICKER_TO_CUSIP = {
    "AAPL": "037833100",  # Apple Inc.
    "MSFT": "594918104",  # Microsoft Corp.
    "GOOGL": "02079K305", # Alphabet Inc. Class A
    "GOOG": "02079K107",  # Alphabet Inc. Class C
    "AMZN": "023135106",  # Amazon.com, Inc.
    "TSLA": "88160R101",  # Tesla, Inc.
    "NVDA": "67066G104",  # NVIDIA Corporation
    "JPM": "46625H100",   # JPMorgan Chase & Co.
    "BAC": "060505104",   # Bank of America Corp
    "XOM": "30231G102",   # Exxon Mobil Corporation
    "CVX": "166764100",   # Chevron Corporation
    "OXY": "693475105",   # Occidental Petroleum Corporation
    # Add more tickers and CUSIPs
}

# Create reverse mapping for display (simplistic - assumes one ticker per CUSIP for demo)
CUSIP_TO_TICKER = {v: k for k, v in TICKER_TO_CUSIP.items()}

# --- SEC EDGAR Interaction Class ---

class SECEdgarHandler:
    """
    Handles interactions with the SEC EDGAR database, including fetching filings,
    finding the 13F Information Table XML, and parsing holdings.
    Adheres to SEC fair access policies.
    """
    BASE_URL = "https://www.sec.gov"
    ARCHIVES_URL = f"{BASE_URL}/Archives/edgar/data"
    SUBMISSIONS_API_URL = f"https://data.sec.gov/submissions/CIK{{cik}}.json" # Corrected endpoint

    def __init__(self, user_agent):
        self.user_agent = user_agent
        self.session = self._create_session()

    def _create_session(self):
        session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        session.headers.update({"User-Agent": self.user_agent})
        return session

    def _make_request(self, url, stream=False):
        time.sleep(SEC_REQUEST_DELAY)
        try:
            response = self.session.get(url, stream=stream)
            response.raise_for_status()
            logging.info(f"Successfully fetched: {url} (Status: {response.status_code})")
            return response
        except requests.exceptions.RequestException as e:
            status_code = e.response.status_code if e.response is not None else "N/A"
            logging.error(f"Request failed for {url} (Status: {status_code}): {e}")
            error_detail = f"Status Code: {status_code}, Reason: {e}"
            st.error(f"Error fetching data from SEC EDGAR: {url}. {error_detail}. Check CIK/Accession# and SEC EDGAR status.")
            return None

    def get_submissions(self, cik):
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
        submissions_data = self.get_submissions(cik)
        if not submissions_data:
             st.warning(f"Could not retrieve submission data for CIK {cik}. Check CIK or SEC API status.")
             return None
        if 'filings' not in submissions_data or 'recent' not in submissions_data['filings']:
            st.warning(f"No recent filings structure found for CIK {cik} in API response.")
            if 'filings' in submissions_data and not submissions_data['filings'].get('recent'):
                 st.info(f"The 'recent' filings list appears empty for CIK {cik}.")
            return None
        recent_filings = submissions_data['filings']['recent']
        required_keys = ['accessionNumber', 'form', 'reportDate', 'filingDate']
        if not all(key in recent_filings for key in required_keys):
             st.warning(f"Submission data for CIK {cik} is missing expected keys (e.g., accessionNumber, form). API structure might have changed.")
             return None
        if not recent_filings.get('accessionNumber'): # Check if list is empty or key missing
             st.warning(f"No recent accession numbers found for CIK {cik}.")
             return None

        target_report_date_str = self._get_quarter_end_date(report_year, report_quarter)
        if not target_report_date_str:
            return None

        latest_matching_filing = None
        latest_filing_date = datetime.min.date()

        try:
            # Check if keys actually contain list-like data before iterating
            if not isinstance(recent_filings.get('accessionNumber'), list):
                 st.warning(f"Unexpected format for filings data for CIK {cik}. Expected lists.")
                 return None

            for i in range(len(recent_filings['accessionNumber'])):
                # Add checks for list index access
                if i >= len(recent_filings['form']) or \
                   i >= len(recent_filings['reportDate']) or \
                   i >= len(recent_filings['filingDate']):
                    logging.warning(f"Inconsistent list lengths in filings data for CIK {cik}. Skipping index {i}.")
                    continue

                form = recent_filings['form'][i]
                if form in ["13F-HR", "13F-HR/A"]:
                    report_date = recent_filings['reportDate'][i]
                    filing_date_str = recent_filings['filingDate'][i]

                    if report_date == target_report_date_str:
                        try:
                            current_filing_date = datetime.strptime(filing_date_str, '%Y-%m-%d').date()
                            if latest_matching_filing is None or current_filing_date >= latest_filing_date:
                                 latest_filing_date = current_filing_date
                                 latest_matching_filing = recent_filings['accessionNumber'][i]
                        except ValueError:
                             logging.warning(f"Could not parse filing date {filing_date_str} for CIK {cik}")
                             continue
        except KeyError as e:
             st.error(f"Missing key '{e}' in filings data for CIK {cik}. API response structure might be unexpected.")
             return None
        except TypeError as e:
             st.error(f"Type error processing filings data for CIK {cik}: {e}. API response structure might be unexpected.")
             return None


        if latest_matching_filing:
            logging.info(f"Found accession {latest_matching_filing} for CIK {cik}, Period {target_report_date_str}")
            return latest_matching_filing
        else:
            st.warning(f"No 13F-HR filing found for CIK {cik} for quarter ending {target_report_date_str}.")
            return None

    def _get_quarter_end_date(self, year, quarter):
        if quarter == 1: return f"{year}-03-31"
        elif quarter == 2: return f"{year}-06-30"
        elif quarter == 3: return f"{year}-09-30"
        elif quarter == 4: return f"{year}-12-31"
        else:
            st.error("Invalid quarter selected.")
            return None

    def _get_primary_doc_url(self, cik, accession_number):
        cik_padded = str(cik).zfill(10)
        accession_no_dash = accession_number.replace('-', '')
        return f"{self.ARCHIVES_URL}/{cik_padded}/{accession_no_dash}/{accession_number}.txt"

    def _get_html_index_url(self, cik, accession_number):
        cik_padded = str(cik).zfill(10)
        accession_no_dash = accession_number.replace('-', '')
        return f"{self.ARCHIVES_URL}/{cik_padded}/{accession_no_dash}/{accession_number}-index.html"

    def _fetch_primary_doc(self, cik, accession_number):
        url = self._get_primary_doc_url(cik, accession_number)
        response = self._make_request(url)
        return response.text if response else None

    def _fetch_html_index(self, cik, accession_number):
        url_html = self._get_html_index_url(cik, accession_number)
        response = self._make_request(url_html)
        if response: return response.text
        else:
            url_htm = url_html.replace("-index.html", "-index.htm")
            logging.info(f"Primary index URL failed ({url_html}), trying fallback: {url_htm}")
            response_htm = self._make_request(url_htm)
            return response_htm.text if response_htm else None

    @st.cache_data(ttl=3600)
    def find_info_table_xml_filename(_self, cik, accession_number):
        logging.info(f"Searching for Info Table XML for {cik} / {accession_number}")
        primary_doc_content = _self._fetch_primary_doc(cik, accession_number)
        if not primary_doc_content: return None

        # Method 1: Parse Manifest
        try:
            parts = re.split(r'</?DOCUMENT>', primary_doc_content, flags=re.IGNORECASE | re.DOTALL)
            document_blocks = parts[1:]
            logging.info(f"Found {len(document_blocks)} potential document blocks in primary doc.")
            for block in document_blocks:
                 if not block.strip(): continue
                 type_match = re.search(r'<TYPE>\s*(.*?)\s*<', block, re.IGNORECASE | re.DOTALL)
                 filename_match = re.search(r'<FILENAME>\s*(.*?)\s*<', block, re.IGNORECASE | re.DOTALL)
                 if type_match and filename_match:
                    doc_type = type_match.group(1).strip().upper()
                    doc_filename = filename_match.group(1).strip()
                    if 'INFORMATION TABLE' in doc_type or 'FORM 13F INFORMATION TABLE' in doc_type:
                        if doc_filename.lower().endswith('.xml'):
                            logging.info(f"Found Info Table via Manifest: Type='{doc_type}', Filename='{doc_filename}'")
                            return doc_filename
                        else: logging.warning(f"Found INFORMATION TABLE type, but filename '{doc_filename}' is not XML. Skipping.")
        except Exception as e: logging.error(f"Error parsing primary document manifest for {accession_number}: {e}")

        # Method 2: Parse HTML Index
        logging.info(f"Manifest parsing failed or yielded no XML. Trying HTML index page for {accession_number}.")
        html_index_content = _self._fetch_html_index(cik, accession_number)
        if html_index_content:
            try:
                soup = BeautifulSoup(html_index_content, 'lxml')
                doc_table = None
                for table in soup.find_all('table'):
                     summary = table.get('summary', '').lower()
                     if 'document format files' in summary or 'submission documents' in summary: doc_table = table; break
                     headers = [th.get_text(strip=True).lower() for th in table.find_all('th')]
                     if 'document' in headers and ('type' in headers or 'description' in headers): doc_table = table
                if doc_table:
                    logging.info("Found potential document table in HTML index.")
                    rows = doc_table.find_all('tr')
                    for row in rows[1:]:
                        cells = row.find_all('td')
                        if len(cells) >= 3:
                            try:
                                doc_link_cell = cells[1]; type_desc_cell = cells[3] if len(cells) > 3 else cells[2]
                                type_desc_text = type_desc_cell.get_text(strip=True).upper()
                                doc_link = doc_link_cell.find('a')
                                if doc_link and doc_link.has_attr('href'):
                                    filename = doc_link.get_text(strip=True); href = doc_link['href']
                                    if 'INFORMATION TABLE' in type_desc_text or '13F TABLE' in type_desc_text:
                                         if filename.lower().endswith('.xml'):
                                             logging.info(f"Found Info Table via HTML Index: Type='{type_desc_text}', Filename='{filename}'"); return filename
                                         elif href.lower().endswith('.xml'):
                                             xml_filename_from_href = href.split('/')[-1]
                                             logging.info(f"Found Info Table XML via HTML Index Href: Type='{type_desc_text}', Filename='{xml_filename_from_href}'"); return xml_filename_from_href
                                         else: logging.warning(f"Found INFORMATION TABLE type '{type_desc_text}' in HTML index, but link '{filename}'/'{href}' doesn't point to XML.")
                            except IndexError: logging.warning("Skipping row in HTML table due to unexpected cell count."); continue
            except Exception as e: logging.error(f"Error parsing HTML index page for {accession_number}: {e}")

        # Method 3: Check Defaults
        logging.warning(f"HTML index parsing failed or yielded no XML. Trying default filenames for {accession_number}.")
        cik_padded = str(cik).zfill(10); accession_no_dash = accession_number.replace('-', '')
        base_path = f"{_self.ARCHIVES_URL}/{cik_padded}/{accession_no_dash}"
        default_filenames = ['infotable.xml', 'form13fInfoTable.xml', 'information_table.xml']
        for filename in default_filenames:
            url = f"{base_path}/{filename}"; time.sleep(SEC_REQUEST_DELAY)
            try:
                response = _self.session.head(url, timeout=5)
                if response.status_code == 200:
                     content_type = response.headers.get('Content-Type', '').lower()
                     if 'xml' in content_type or 'text/plain' in content_type:
                          logging.info(f"Found Info Table via Default Filename Check: '{filename}' (Content-Type: {content_type})"); return filename
                     else: logging.warning(f"Default filename '{filename}' found but Content-Type is '{content_type}'.")
            except requests.exceptions.RequestException as e: logging.debug(f"HEAD request failed for default {filename}: {e}"); continue

        logging.error(f"Could not identify Information Table XML filename for {accession_number} using any method.")
        st.error(f"Failed to find the Information Table XML file within filing {accession_number}.")
        return None

    def _get_info_table_xml_url(self, cik, accession_number, xml_filename):
        cik_padded = str(cik).zfill(10); accession_no_dash = accession_number.replace('-', '')
        xml_filename_cleaned = xml_filename.lstrip('/')
        return f"{self.ARCHIVES_URL}/{cik_padded}/{accession_no_dash}/{xml_filename_cleaned}"

    def _fetch_info_table_xml(self, xml_url):
        response = self._make_request(xml_url)
        return response.content if response else None

    def parse_info_table_xml(self, xml_content):
        if not xml_content: return pd.DataFrame()
        try:
            xml_file = BytesIO(xml_content)
            parser = etree.XMLParser(remove_pis=True, recover=True)
            tree = etree.parse(xml_file, parser=parser)
            root = tree.getroot()
            ns = {}; default_uri = root.nsmap.get(None) if hasattr(root, 'nsmap') and root.nsmap else None
            if default_uri: ns['dflt'] = default_uri; logging.info(f"Detected default namespace: {default_uri}")
            common_table_ns_uri = 'http://www.sec.gov/edgar/document/thirteenf/informationtable'
            if common_table_ns_uri not in ns.values(): ns['tns'] = common_table_ns_uri; logging.info(f"Adding common 'tns' prefix")
            if hasattr(root, 'nsmap'):
                 for prefix, uri in root.nsmap.items():
                      if prefix and prefix not in ns: ns[prefix] = uri; logging.info(f"Adding explicit prefix '{prefix}'")

            infotable_paths = [];
            if 'dflt' in ns: infotable_paths.append('.//dflt:infoTable')
            if 'tns' in ns: infotable_paths.append('.//tns:infoTable')
            infotable_paths.append('.//infoTable')
            infotable_xpath = " | ".join(infotable_paths) + " | .//*[local-name()='infoTable']"
            logging.info(f"Using XPath for infoTable: {infotable_xpath}")
            info_tables = root.xpath(infotable_xpath, namespaces=ns)

            if not info_tables:
                 root_tag_local = etree.QName(root.tag).localname
                 if root_tag_local == 'infoTable': info_tables = [root]; logging.info("Root element is infoTable.")
                 else: logging.warning(f"Could not find 'infoTable' elements using XPath."); return pd.DataFrame()

            logging.info(f"Found {len(info_tables)} infoTable elements.")
            holdings = []

            def get_text_xpath(element, tag_name):
                paths_to_try = []; prefixes = ['dflt', 'tns'] + [p for p in ns if p not in ['dflt', 'tns']]
                for prefix in prefixes: paths_to_try.append(f'.//{prefix}:{tag_name}')
                paths_to_try.append(f'.//{tag_name}'); paths_to_try.append(f'.//*[local-name()="{tag_name}"]')
                for path in paths_to_try:
                    try:
                        results = element.findall(path, namespaces=ns)
                        if not results and '*' in path: results = element.xpath(path, namespaces=ns)
                        for res in results:
                             if res is not None and res.text is not None and res.text.strip(): return res.text.strip()
                    except Exception: continue
                # logging.warning(f"Could not find text for tag '{tag_name}'")
                return None

            def get_nested_text_xpath(element, path_tags):
                paths_to_try = []; prefixes = ['dflt', 'tns', ''] # '' for no prefix
                # Simple patterns
                for pre in prefixes:
                     current_path = ".//" + "/".join([f"{pre}:{t}" if pre else t for t in path_tags])
                     paths_to_try.append(current_path)
                # Local-name fallback
                paths_to_try.append( ".//" + "/".join([f'*[local-name()="{t}"]' for t in path_tags]) )

                for path in paths_to_try:
                    try:
                        results = element.xpath(path, namespaces=ns)
                        if results and results[0].text is not None and results[0].text.strip(): return results[0].text.strip()
                    except Exception: continue
                # logging.warning(f"Could not find nested text for path '{'/'.join(path_tags)}'")
                return None

            for i, table in enumerate(info_tables):
                holding = {}
                try:
                    holding['nameOfIssuer'] = get_text_xpath(table, 'nameOfIssuer')
                    holding['titleOfClass'] = get_text_xpath(table, 'titleOfClass')
                    holding['cusip'] = get_text_xpath(table, 'cusip')
                    value_text = get_text_xpath(table, 'value')
                    value_cleaned = value_text.replace(',', '').strip() if value_text else ''
                    holding['value'] = int(value_cleaned) * 1000 if value_cleaned.isdigit() else 0

                    ssh_prnamt = get_nested_text_xpath(table, ['shrsOrPrnAmt', 'sshPrnamt']) or get_text_xpath(table, 'sshPrnamt')
                    ssh_prnamt_type = get_nested_text_xpath(table, ['shrsOrPrnAmt', 'sshPrnamtType']) or get_text_xpath(table, 'sshPrnamtType')
                    ssh_cleaned = ssh_prnamt.replace(',', '').strip() if ssh_prnamt else ''
                    holding['sshPrnamt'] = int(ssh_cleaned) if ssh_cleaned.isdigit() else 0
                    holding['sshPrnamtType'] = ssh_prnamt_type

                    holding['investmentDiscretion'] = get_text_xpath(table, 'investmentDiscretion')

                    sole_auth = get_nested_text_xpath(table, ['votingAuthority', 'Sole'])
                    shared_auth = get_nested_text_xpath(table, ['votingAuthority', 'Shared'])
                    none_auth = get_nested_text_xpath(table, ['votingAuthority', 'None'])
                    holding['votingAuthSole'] = int(sole_auth.replace(',', '').strip()) if sole_auth and sole_auth.replace(',', '').strip().isdigit() else 0
                    holding['votingAuthShared'] = int(shared_auth.replace(',', '').strip()) if shared_auth and shared_auth.replace(',', '').strip().isdigit() else 0
                    holding['votingAuthNone'] = int(none_auth.replace(',', '').strip()) if none_auth and none_auth.replace(',', '').strip().isdigit() else 0

                    if holding['cusip']:
                        cusip_cleaned = holding['cusip'].strip().upper().replace(' ','') # Remove spaces too
                        if len(cusip_cleaned) > 9: cusip_cleaned = cusip_cleaned[:9]; logging.warning(f"Corrected CUSIP length > 9")
                        if len(cusip_cleaned) < 8: logging.warning(f"Skipping holding due to short CUSIP '{cusip_cleaned}'"); continue
                        holding['cusip'] = cusip_cleaned; holdings.append(holding)
                    else: logging.warning(f"Skipping holding record #{i+1} due to missing CUSIP")
                except Exception as e: logging.error(f"Error processing holding record #{i+1}: {e}", exc_info=True); continue

            if not holdings: logging.warning("No valid holdings extracted."); return pd.DataFrame()
            df = pd.DataFrame(holdings)
            df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(0).astype(float)
            df['sshPrnamt'] = pd.to_numeric(df['sshPrnamt'], errors='coerce').fillna(0).astype(int)
            df['votingAuthSole'] = pd.to_numeric(df['votingAuthSole'], errors='coerce').fillna(0).astype(int)
            df['votingAuthShared'] = pd.to_numeric(df['votingAuthShared'], errors='coerce').fillna(0).astype(int)
            df['votingAuthNone'] = pd.to_numeric(df['votingAuthNone'], errors='coerce').fillna(0).astype(int)
            df['cusip'] = df['cusip'].astype(str).str.strip().str.upper()
            str_cols = ['nameOfIssuer', 'titleOfClass', 'sshPrnamtType', 'investmentDiscretion']
            for col in str_cols:
                 if col in df.columns: df[col] = df[col].fillna('').astype(str).str.replace(r'[^\x20-\x7E]+', '', regex=True).str.strip()
            logging.info(f"Successfully parsed and cleaned {len(df)} holdings from XML.")
            return df

        except etree.XMLSyntaxError as e:
            line = getattr(e, 'lineno', 'N/A'); pos = getattr(e, 'position', 'N/A'); msg = getattr(e, 'msg', str(e))
            logging.error(f"XML Syntax Error parsing info table (Line: {line}, Pos: {pos}): {msg}")
            st.error(f"Failed to parse XML (Line: {line}). Error: {msg}")
            return pd.DataFrame() # Return empty df on syntax error
        except Exception as e:
            logging.error(f"Unexpected error during XML parsing or DataFrame creation: {e}", exc_info=True)
            st.error(f"An unexpected error occurred during XML parsing: {e}")
            return pd.DataFrame() # Return empty df on other errors


# --- Data Processing Functions ---

@st.cache_data(ttl=3600)
def get_holdings_data(cik, year, quarter):
    """ Orchestrates fetching/parsing. Returns DataFrame or None on critical failure. """
    handler = SECEdgarHandler(SEC_USER_AGENT)
    accession_number = handler.find_latest_13f_hr_accession(cik, year, quarter)
    if not accession_number: return None
    xml_filename = handler.find_info_table_xml_filename(cik, accession_number)
    if not xml_filename: return None
    xml_url = handler._get_info_table_xml_url(cik, accession_number, xml_filename)
    xml_content = handler._fetch_info_table_xml(xml_url)
    if not xml_content: st.error(f"Failed to download XML: {xml_url}"); return None
    parsed_df = handler.parse_info_table_xml(xml_content)
    # Return the DataFrame (could be empty if parsing found no valid holdings)
    return parsed_df


def get_previous_quarter(year, quarter):
    if quarter == 1: return year - 1, 4
    else: return year, quarter - 1

@st.cache_data(ttl=3600)
def calculate_holding_changes(_df_curr, _df_prev):
    """ Compares holdings between two quarters (DataFrames), handles None inputs. """
    if _df_curr is None: logging.error("Current holdings data is None."); return pd.DataFrame()
    if _df_prev is None: logging.warning("Previous holdings data is None."); _df_prev = pd.DataFrame()

    required_cols = ['cusip', 'nameOfIssuer', 'sshPrnamt', 'value']
    if not all(col in _df_curr.columns for col in required_cols):
         logging.error("Current DF missing required columns.")
         # Try to return something useful, even if incomplete
         _df_curr['change_type'] = 'Error (Missing Cols)'
         for col in required_cols + ['change_shares', 'change_value', 'change_pct']:
              if col not in _df_curr.columns: _df_curr[col] = pd.NA
         return _df_curr.reindex(columns=['cusip', 'nameOfIssuer', 'sshPrnamt', 'value', 'change_type', 'change_shares', 'change_value', 'change_pct'], fill_value=pd.NA)

    if not _df_prev.empty and not all(col in _df_prev.columns for col in required_cols):
          logging.warning("Previous DF exists but missing required columns. Treating as empty.")
          _df_prev = pd.DataFrame(columns=required_cols)

    df_curr_copy = _df_curr.copy()
    df_prev_copy = _df_prev.copy()

    if df_prev_copy.empty:
        df_curr_copy['change_type'] = 'New'; df_curr_copy['change_shares'] = df_curr_copy['sshPrnamt']
        df_curr_copy['change_value'] = df_curr_copy['value']; df_curr_copy['change_pct'] = 100.0
        return df_curr_copy[['cusip', 'nameOfIssuer', 'sshPrnamt', 'value', 'change_type', 'change_shares', 'change_value', 'change_pct']]

    df_curr_agg = df_curr_copy.groupby('cusip').agg(
        nameOfIssuer=('nameOfIssuer', lambda x: x.dropna().iloc[0] if not x.dropna().empty else x.iloc[0]),
        sshPrnamt=('sshPrnamt', 'sum'), value=('value', 'sum')
    ).reset_index()
    df_prev_agg = df_prev_copy.groupby('cusip').agg(
        nameOfIssuer=('nameOfIssuer', lambda x: x.dropna().iloc[0] if not x.dropna().empty else x.iloc[0]),
        sshPrnamt=('sshPrnamt', 'sum'), value=('value', 'sum')
    ).reset_index()

    df_merged = pd.merge(df_curr_agg, df_prev_agg, on='cusip', how='outer', suffixes=('_curr', '_prev'))
    changes = []
    for _, row in df_merged.iterrows():
        cusip = row['cusip']; name = row['nameOfIssuer_curr'] if pd.notna(row['nameOfIssuer_curr']) else row['nameOfIssuer_prev']
        shares_curr = int(row['sshPrnamt_curr']) if pd.notna(row['sshPrnamt_curr']) else 0
        value_curr = float(row['value_curr']) if pd.notna(row['value_curr']) else 0.0
        shares_prev = int(row['sshPrnamt_prev']) if pd.notna(row['sshPrnamt_prev']) else 0
        value_prev = float(row['value_prev']) if pd.notna(row['value_prev']) else 0.0
        change_type = ''; change_shares = 0; change_value = 0.0; change_pct = pd.NA
        is_new = shares_curr > 0 and shares_prev == 0; is_exited = shares_curr == 0 and shares_prev > 0
        is_held_both = shares_curr > 0 and shares_prev > 0
        if is_new: change_type = 'New'; change_shares = shares_curr; change_value = value_curr; change_pct = 100.0
        elif is_exited: change_type = 'Exited'; change_shares = -shares_prev; change_value = -value_prev; change_pct = -100.0
        elif is_held_both:
            change_shares = shares_curr - shares_prev; change_value = value_curr - value_prev
            if shares_prev != 0: change_pct = (shares_curr - shares_prev) / shares_prev * 100.0
            if change_shares > 0: change_type = 'Increased'
            elif change_shares < 0: change_type = 'Decreased'
            else:
                 if abs(value_curr - value_prev) > 1: change_type = 'Value Change Only'
                 else: change_type = 'Unchanged'; change_pct = 0.0
        else: continue # Skip rows where holding was zero in both periods
        changes.append({ 'cusip': cusip, 'nameOfIssuer': name, 'sshPrnamt': shares_curr, 'value': value_curr, 'change_type': change_type, 'change_shares': change_shares, 'change_value': change_value, 'change_pct': change_pct })

    if not changes: return pd.DataFrame(columns=['cusip', 'nameOfIssuer', 'sshPrnamt', 'value', 'change_type', 'change_shares', 'change_value', 'change_pct'])
    df_changes = pd.DataFrame(changes)
    df_changes['value'] = pd.to_numeric(df_changes['value'], errors='coerce').fillna(0).astype(float)
    df_changes['sshPrnamt'] = pd.to_numeric(df_changes['sshPrnamt'], errors='coerce').fillna(0).astype(int)
    df_changes['change_shares'] = pd.to_numeric(df_changes['change_shares'], errors='coerce').fillna(0).astype(int)
    df_changes['change_value'] = pd.to_numeric(df_changes['change_value'], errors='coerce').fillna(0).astype(float)
    df_changes['change_pct'] = pd.to_numeric(df_changes['change_pct'], errors='coerce').fillna(pd.NA)
    return df_changes

def add_ticker_column(df, cusip_to_ticker_map):
    """Adds a 'Ticker' column based on CUSIP mapping."""
    if 'cusip' in df.columns:
        df['Ticker'] = df['cusip'].map(cusip_to_ticker_map).fillna('N/A')
        # Reorder columns to put Ticker near the beginning
        cols = df.columns.tolist()
        if 'Ticker' in cols:
            cols.insert(cols.index('cusip') + 1, cols.pop(cols.index('Ticker')))
            df = df[cols]
    return df


# --- Streamlit App UI ---

st.set_page_config(layout="wide", page_title="13F Holdings Tracker")

st.title("📈 SEC Form 13F Institutional Holdings Tracker")
st.caption(f"Using SEC EDGAR data | User Agent: {SEC_USER_AGENT}")

st.sidebar.header("Query Parameters")

# --- Fund Selection ---
available_funds = list(FUND_NAME_TO_CIK.keys())
selected_fund_names = st.sidebar.multiselect(
    "Select Investment Manager(s)",
    options=available_funds,
    default=available_funds[:2], # Default to first two funds
    help="Select one or more managers from the list."
)

# Convert selected names to CIKs
selected_ciks = [FUND_NAME_TO_CIK[name] for name in selected_fund_names if name in FUND_NAME_TO_CIK]

# --- Quarter Selection ---
current_year = datetime.now().year
years = list(range(current_year, current_year - 10, -1)) # Last 10 years
quarters = [1, 2, 3, 4]
today = datetime.now().date(); month = today.month
default_q = 4; default_year = current_year - 1
if month >= 5 and month < 8: default_q = 1; default_year = current_year
elif month >= 8 and month < 11: default_q = 2; default_year = current_year
elif month >= 11: default_q = 3; default_year = current_year
elif month >= 2 and month < 5: default_q = 4; default_year = current_year -1
# Else default is Q4 prev year (for Jan)

try: default_year_index = years.index(default_year)
except ValueError: default_year_index = 0
selected_year = st.sidebar.selectbox("Year", years, index=default_year_index)
selected_quarter = st.sidebar.selectbox("Quarter", quarters, index=default_q - 1)

# --- View Selection ---
view_options = ["Manager View", "Stock View"]
selected_view = st.sidebar.radio("Select View", view_options)

# --- Stock Selection (for Stock View) ---
selected_ticker = ""
selected_cusip = None
if selected_view == "Stock View":
    available_tickers = list(TICKER_TO_CUSIP.keys())
    selected_ticker = st.sidebar.selectbox(
        "Select Stock Ticker",
        options=[""] + available_tickers, # Add empty option
        index=0, # Default to empty
        help="Select a stock ticker from the list."
    )
    if selected_ticker and selected_ticker in TICKER_TO_CUSIP:
        selected_cusip = TICKER_TO_CUSIP[selected_ticker]
        st.sidebar.caption(f"Maps to CUSIP: {selected_cusip}")
    elif selected_ticker:
        st.sidebar.warning(f"Ticker '{selected_ticker}' not found in predefined map.")


st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Note:** Fund/Stock selection is limited to predefined examples in this demo version.
    **Limitations:** Data lag (45d+), limited security types, CIK/CUSIP mapping accuracy.
    """
)

# --- Main Application Logic ---

def load_manager_data(cik_list, year, quarter):
    prev_year, prev_quarter = get_previous_quarter(year, quarter)
    loaded_data = {}
    total_ciks = len(cik_list)
    with st.status(f"Loading data for {total_ciks} manager(s)...", expanded=False) as status:
        for i, cik in enumerate(cik_list):
            # Find fund name from CIK for display
            fund_name = next((name for name, c in FUND_NAME_TO_CIK.items() if c == cik), f"CIK {cik}")
            status.update(label=f"({i+1}/{total_ciks}) Loading: {fund_name} (CIK: {cik})...")
            st.write(f"({i+1}/{total_ciks}) Loading data for: **{fund_name}** (CIK: {cik})")

            df_curr = get_holdings_data(cik, year, quarter)
            if df_curr is not None: st.write(f"-> Q{quarter} {year} holdings: {len(df_curr)} records.")
            else: st.write(f"-> Q{quarter} {year} holdings: Failed to load/parse.")

            df_prev = get_holdings_data(cik, prev_year, prev_quarter)
            if df_prev is not None: st.write(f"-> Q{prev_quarter} {prev_year} holdings: {len(df_prev)} records.")
            else: st.write(f"-> Q{prev_quarter} {prev_year} holdings: Failed to load/parse.")

            loaded_data[cik] = {'current': df_curr, 'previous': df_prev}
        status.update(label="Data loading complete!", state="complete", expanded=False)
    return loaded_data

# --- Display Logic ---
if not selected_ciks:
    st.warning("Please select at least one Investment Manager from the sidebar.")
else:
    all_manager_data = load_manager_data(selected_ciks, selected_year, selected_quarter)

    if selected_view == "Manager View":
        st.header(f"Manager View - {selected_year} Q{selected_quarter}")
        # Allow selecting which loaded Fund Name to view
        fund_names_with_data = [name for name, cik in FUND_NAME_TO_CIK.items() if cik in selected_ciks]
        if fund_names_with_data:
            selected_name_to_display = st.selectbox("Select Manager to Display", fund_names_with_data)
            selected_cik_to_display = FUND_NAME_TO_CIK.get(selected_name_to_display)
        else:
            selected_cik_to_display = None

        if selected_cik_to_display and selected_cik_to_display in all_manager_data:
            data = all_manager_data[selected_cik_to_display]
            df_curr = data['current']
            df_prev = data['previous']

            if df_curr is None:
                st.error(f"Could not load/parse current quarter data for {selected_name_to_display}. See warnings above.")
            elif df_curr.empty:
                 st.info(f"No holdings data found for {selected_name_to_display} for Q{selected_quarter} {selected_year}.")
            else:
                 prev_yr, prev_q = get_previous_quarter(selected_year, selected_quarter)
                 st.subheader(f"Holdings Changes for: **{selected_name_to_display}** (Q{selected_quarter} {selected_year} vs Q{prev_q} {prev_yr})")

                 if df_prev is None:
                      st.warning(f"Could not load previous quarter data. Changes displayed assuming all current holdings are new.")
                      df_changes = calculate_holding_changes(df_curr, None)
                 else:
                      df_changes = calculate_holding_changes(df_curr, df_prev)

                 if not df_changes.empty:
                     st.metric("Total Holdings Reported (Current)", len(df_curr))
                     if 'change_type' in df_changes.columns: change_counts = df_changes['change_type'].value_counts()
                     else: change_counts = pd.Series(dtype=int)

                     col1, col2, col3, col4, col5 = st.columns(5)
                     col1.metric("New", change_counts.get('New', 0)); col2.metric("Increased", change_counts.get('Increased', 0))
                     col3.metric("Decreased", change_counts.get('Decreased', 0)); col4.metric("Exited", change_counts.get('Exited', 0))
                     col5.metric("Unchanged/Value", change_counts.get('Unchanged', 0) + change_counts.get('Value Change Only', 0))

                     # Add Ticker column for display
                     df_changes_with_ticker = add_ticker_column(df_changes, CUSIP_TO_TICKER)

                     df_display = df_changes_with_ticker[[
                         'cusip', 'Ticker', 'nameOfIssuer', 'change_type',
                         'sshPrnamt', 'value',
                         'change_shares', 'change_value', 'change_pct'
                     ]].sort_values(by=['change_type', 'value'], ascending=[True, False]).reset_index(drop=True)

                     st.dataframe(df_display.style.format({
                         'value': "${:,.0f}", 'change_shares': "{:+,}", 'change_value': "${:+,}", 'change_pct': "{:.1f}%"
                     }).format(precision=1, na_rep='N/A', subset=['change_pct']), use_container_width=True)

                 elif df_curr is not None and not df_curr.empty:
                      st.info(f"No changes detected or previous data unavailable/identical. Displaying current holdings.")
                      df_curr_with_ticker = add_ticker_column(df_curr, CUSIP_TO_TICKER)
                      st.dataframe(df_curr_with_ticker[['cusip', 'Ticker', 'nameOfIssuer', 'sshPrnamt', 'value']].style.format({'value': "${:,.0f}"}), use_container_width=True)


    elif selected_view == "Stock View":
        st.header(f"Stock View - {selected_year} Q{selected_quarter}")

        if not selected_ticker or selected_cusip is None:
            st.warning("Please select a valid Stock Ticker from the sidebar for the Stock View.")
        elif not all_manager_data:
             st.warning("No manager data loaded. Please select managers and check quarter.")
        else:
            st.subheader(f"Activity in Stock: **{selected_ticker}** (CUSIP: {selected_cusip}) by Selected Managers")
            stock_activity = []
            stock_name_display = selected_ticker # Start with ticker

            for cik, data in all_manager_data.items():
                 df_curr = data['current']
                 df_prev = data['previous']
                 fund_name = next((name for name, c in FUND_NAME_TO_CIK.items() if c == cik), f"CIK {cik}")

                 if df_curr is None: logging.warning(f"Skipping CIK {cik} for stock view (missing current data)."); continue

                 df_changes = calculate_holding_changes(df_curr, df_prev) # Handles df_prev being None

                 stock_row = df_changes[df_changes['cusip'] == selected_cusip]

                 if not stock_row.empty:
                     activity = stock_row.iloc[0].to_dict()
                     activity['cik'] = cik
                     activity['fundName'] = fund_name # Add fund name
                     stock_activity.append(activity)
                     # Try get official name from data if available
                     if stock_name_display == selected_ticker and pd.notna(activity['nameOfIssuer']):
                         stock_name_display = activity['nameOfIssuer']

            if stock_activity:
                st.subheader(f"Activity for {stock_name_display} ({selected_ticker} / {selected_cusip})")
                df_stock_view = pd.DataFrame(stock_activity)
                # Add Ticker column - should match selected_ticker but good practice
                df_stock_view_with_ticker = add_ticker_column(df_stock_view, CUSIP_TO_TICKER)

                df_stock_display = df_stock_view_with_ticker[[
                     'fundName', 'cik', 'cusip', 'Ticker', 'nameOfIssuer', 'change_type',
                     'sshPrnamt', 'value', 'change_shares', 'change_value', 'change_pct'
                ]].sort_values(by=['change_type', 'value'], ascending=[True, False]).reset_index(drop=True)

                st.dataframe(df_stock_display.style.format({
                     'value': "${:,.0f}", 'change_shares': "{:+,}", 'change_value': "${:+,}",'change_pct': "{:.1f}%"
                }).format(precision=1, na_rep='N/A', subset=['change_pct']), use_container_width=True)
            else:
                st.info(f"No activity found for {selected_ticker} (CUSIP: {selected_cusip}) among the selected managers for the period.")
