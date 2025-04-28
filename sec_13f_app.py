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
    def __init__(self, user_agent): # <<< Ensure 4 spaces indentation
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
            # Using re.split is sometimes more robust than re.findall for nested/malformed structures
            parts = re.split(r'</?DOCUMENT>', primary_doc_content, flags=re.IGNORECASE | re.DOTALL)
            document_blocks = parts[1:] # Get content potentially within <DOCUMENT> tags
            logging.info(f"Found {len(document_blocks)} potential document blocks in primary doc.")

            for block in document_blocks:
                 if not block.strip(): continue # Skip empty blocks from split
                 # More robust regex to handle potential extra whitespace/newlines within tags
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
                                    href = doc_link['href']
                                    # Check if type/description indicates information table
                                    if 'INFORMATION TABLE' in type_desc_text or '13F TABLE' in type_desc_text:
                                         if filename.lower().endswith('.xml'):
                                             logging.info(f"Found Info Table via HTML Index: Type='{type_desc_text}', Filename='{filename}'")
                                             return filename
                                         # Check if href points to XML even if text doesn't say .xml
                                         elif href.lower().endswith('.xml'):
                                             xml_filename_from_href = href.split('/')[-1]
                                             logging.info(f"Found Info Table XML via HTML Index Href: Type='{type_desc_text}', Href='{href}', Filename='{xml_filename_from_href}'")
                                             return xml_filename_from_href
                                         else:
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
                     if 'xml' in content_type or 'text/plain' in content_type: # Accept text/plain too just in case
                          logging.info(f"Found Info Table via Default Filename Check: '{filename}' (Content-Type: {content_type})")
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

    # ========================================================================
    # REVISED parse_info_table_xml method starts here
    # ========================================================================
    def parse_info_table_xml(self, xml_content):
        """Parses the 13F Information Table XML into a pandas DataFrame."""
        if not xml_content:
            return pd.DataFrame() # Return empty DataFrame if no content

        try:
            # Use BytesIO to handle the XML content
            xml_file = BytesIO(xml_content)
            # Try parsing, remove processing instructions which can cause issues
            # Add recover=True to handle minor XML errors if possible
            parser = etree.XMLParser(remove_pis=True, recover=True)
            tree = etree.parse(xml_file, parser=parser)
            root = tree.getroot()

            # --- Namespace Handling ---
            ns = {}
            # Check for a default namespace (xmlns="...") defined on the root element
            # Use nsmap attribute available on elements
            if root.nsmap:
                default_uri = root.nsmap.get(None)
                if default_uri:
                    ns['dflt'] = default_uri # Assign 'dflt' prefix to the default namespace URI
                    logging.info(f"Detected default namespace: {default_uri}")

            # Define common 13F table namespace URI
            common_table_ns_uri = 'http://www.sec.gov/edgar/document/thirteenf/informationtable'
            # Add common prefix 'tns' if this URI isn't the default or already mapped
            # Check against ns.values() to see if the URI is already mapped
            if common_table_ns_uri not in ns.values():
                 ns['tns'] = common_table_ns_uri
                 logging.info(f"Adding common table namespace prefix 'tns' for URI: {common_table_ns_uri}")

            # Copy other explicit prefixes from the root element's nsmap if needed
            if hasattr(root, 'nsmap'):
                 for prefix, uri in root.nsmap.items():
                      if prefix and prefix not in ns: # Avoid overwriting 'dflt' or 'tns' if keys clash
                           ns[prefix] = uri
                           logging.info(f"Adding explicit namespace prefix '{prefix}' for URI: {uri}")

            # --- Find InfoTable elements ---
            # Construct XPath to find infoTable using potential prefixes OR local-name()
            infotable_paths = []
            if 'dflt' in ns: infotable_paths.append('.//dflt:infoTable')
            if 'tns' in ns: infotable_paths.append('.//tns:infoTable')
            infotable_paths.append('.//infoTable') # No namespace case
            # Add local-name() fallback directly in the main query
            infotable_xpath = " | ".join(infotable_paths) + " | .//*[local-name()='infoTable']"
            logging.info(f"Using XPath for infoTable: {infotable_xpath}")

            info_tables = root.xpath(infotable_xpath, namespaces=ns)

            if not info_tables:
                 # Check if the root element itself is the infoTable (less common but possible)
                 root_tag_local = etree.QName(root.tag).localname
                 if root_tag_local == 'infoTable':
                      info_tables = [root]
                      logging.info("Root element itself is infoTable.")
                 else:
                      logging.warning(f"Could not find any 'infoTable' elements using XPath: {infotable_xpath}")
                      st.error("Failed to find 'infoTable' elements in the XML structure.")
                      return pd.DataFrame()


            logging.info(f"Found {len(info_tables)} infoTable elements.")
            holdings = []

            # Helper to get text using appropriate prefix, trying multiple paths including local-name()
            def get_text_xpath(element, tag_name):
                paths_to_try = []
                # Prioritize known/detected prefixes
                if 'dflt' in ns: paths_to_try.append(f'.//dflt:{tag_name}')
                if 'tns' in ns: paths_to_try.append(f'.//tns:{tag_name}')
                # Try other explicit prefixes found in the document
                for prefix in ns:
                     if prefix not in ['dflt', 'tns']:
                          paths_to_try.append(f'.//{prefix}:{tag_name}')
                # Fallback to no namespace prefix
                paths_to_try.append(f'.//{tag_name}')
                # Fallback to local-name()
                paths_to_try.append(f'.//*[local-name()="{tag_name}"]')

                for path in paths_to_try:
                    try:
                        # Use findall which is sometimes more forgiving than full xpath
                        results = element.findall(path, namespaces=ns)
                        # If findall fails or path is local-name(), try xpath
                        if not results and '*' in path:
                             results = element.xpath(path, namespaces=ns)

                        # Process results: get text from the first non-empty result
                        for res in results:
                             if res is not None and res.text is not None and res.text.strip():
                                  # logging.debug(f"Found text for '{tag_name}' using path: {path}")
                                  return res.text.strip()
                    except (etree.XPathEvalError, etree.XPathSyntaxError, TypeError) as e:
                        # Ignore path if prefix is invalid or other XPath error
                        logging.debug(f"XPath/Find error for tag '{tag_name}' with path '{path}': {e}")
                        continue

                logging.warning(f"Could not find text for tag '{tag_name}' using any path.")
                return None # Return None if not found

            # Helper for nested structures like shrsOrPrnAmt/sshPrnamt using similar logic
            def get_nested_text_xpath(element, path_tags):
                # Simple implementation: try combinations of prefixes/no prefix for each tag
                # This is complex to get perfectly general, focus on common patterns

                # Pattern 1: All default namespace
                if 'dflt' in ns:
                    path1 = ".//" + "/".join([f"dflt:{t}" for t in path_tags])
                    results1 = element.xpath(path1, namespaces=ns)
                    if results1 and results1[0].text is not None and results1[0].text.strip():
                        return results1[0].text.strip()

                # Pattern 2: All common 'tns' namespace
                if 'tns' in ns:
                    path2 = ".//" + "/".join([f"tns:{t}" for t in path_tags])
                    results2 = element.xpath(path2, namespaces=ns)
                    if results2 and results2[0].text is not None and results2[0].text.strip():
                        return results2[0].text.strip()

                 # Pattern 3: No namespace
                path3 = ".//" + "/".join(path_tags)
                results3 = element.xpath(path3, namespaces=ns) # ns might be needed if prefixes used elsewhere
                if results3 and results3[0].text is not None and results3[0].text.strip():
                    return results3[0].text.strip()

                # Pattern 4: local-name() fallback
                path4 = ".//" + "/".join([f'*[local-name()="{t}"]' for t in path_tags])
                results4 = element.xpath(path4) # No ns needed for local-name() generally
                if results4 and results4[0].text is not None and results4[0].text.strip():
                     logging.debug(f"Used local-name() fallback for nested path '{'/'.join(path_tags)}'")
                     return results4[0].text.strip()


                logging.warning(f"Could not find nested text for path '{'/'.join(path_tags)}' using common patterns.")
                return None


            for i, table in enumerate(info_tables):
                holding = {}
                try:
                    holding['nameOfIssuer'] = get_text_xpath(table, 'nameOfIssuer')
                    holding['titleOfClass'] = get_text_xpath(table, 'titleOfClass')
                    holding['cusip'] = get_text_xpath(table, 'cusip')
                    value_text = get_text_xpath(table, 'value')
                    # Handle potential non-numeric issues before conversion
                    if value_text:
                         value_cleaned = value_text.replace(',', '').strip()
                         holding['value'] = int(value_cleaned) * 1000 if value_cleaned.isdigit() else 0
                    else:
                         holding['value'] = 0

                    # Try nested structure first, then direct access for shrsOrPrnAmt fields
                    ssh_prnamt = get_nested_text_xpath(table, ['shrsOrPrnAmt', 'sshPrnamt'])
                    if ssh_prnamt is None: ssh_prnamt = get_text_xpath(table, 'sshPrnamt')

                    ssh_prnamt_type = get_nested_text_xpath(table, ['shrsOrPrnAmt', 'sshPrnamtType'])
                    if ssh_prnamt_type is None: ssh_prnamt_type = get_text_xpath(table, 'sshPrnamtType')

                    # Handle potential non-numeric issues before conversion
                    if ssh_prnamt:
                         ssh_cleaned = ssh_prnamt.replace(',', '').strip()
                         holding['sshPrnamt'] = int(ssh_cleaned) if ssh_cleaned.isdigit() else 0
                    else:
                         holding['sshPrnamt'] = 0
                    holding['sshPrnamtType'] = ssh_prnamt_type

                    holding['investmentDiscretion'] = get_text_xpath(table, 'investmentDiscretion')

                    # Try nested structure first for votingAuthority
                    sole_auth = get_nested_text_xpath(table, ['votingAuthority', 'Sole'])
                    shared_auth = get_nested_text_xpath(table, ['votingAuthority', 'Shared'])
                    none_auth = get_nested_text_xpath(table, ['votingAuthority', 'None'])

                    # Handle potential non-numeric issues before conversion
                    holding['votingAuthSole'] = int(sole_auth.replace(',', '').strip()) if sole_auth and sole_auth.replace(',', '').strip().isdigit() else 0
                    holding['votingAuthShared'] = int(shared_auth.replace(',', '').strip()) if shared_auth and shared_auth.replace(',', '').strip().isdigit() else 0
                    holding['votingAuthNone'] = int(none_auth.replace(',', '').strip()) if none_auth and none_auth.replace(',', '').strip().isdigit() else 0

                    # Basic validation check - require CUSIP
                    if holding['cusip']:
                        # CUSIP validation/cleaning
                        cusip_cleaned = holding['cusip'].strip().upper()
                        if len(cusip_cleaned) > 9:
                             logging.warning(f"Correcting CUSIP length > 9: '{cusip_cleaned}' to '{cusip_cleaned[:9]}' for {holding.get('nameOfIssuer')}")
                             cusip_cleaned = cusip_cleaned[:9]
                        # Allow slightly short CUSIPs if needed, or enforce 9 exactly
                        if len(cusip_cleaned) < 8: # Allow 8 or 9? Let's be strict for now.
                              logging.warning(f"Skipping holding record due to invalid CUSIP length '{cusip_cleaned}' for {holding.get('nameOfIssuer')}")
                              continue # Skip invalid CUSIP
                        holding['cusip'] = cusip_cleaned # Store cleaned CUSIP
                        holdings.append(holding)
                    else:
                        logging.warning(f"Skipping holding record #{i+1} due to missing CUSIP: Name='{holding.get('nameOfIssuer')}'")

                except Exception as e:
                     logging.error(f"Error processing holding record #{i+1}: {e}. Record data: {holding}", exc_info=True)
                     # Optionally skip this record or handle differently
                     continue


            if not holdings:
                 logging.warning("No valid holdings extracted after processing infoTable elements.")
                 # Don't show streamlit error here, let calling function handle empty df
                 # st.warning("Parsed the XML file, but found no valid holding records.")
                 return pd.DataFrame()

            df = pd.DataFrame(holdings)
            # Perform cleaning after creating the DataFrame
            df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(0).astype(float)
            df['sshPrnamt'] = pd.to_numeric(df['sshPrnamt'], errors='coerce').fillna(0).astype(int)
            df['votingAuthSole'] = pd.to_numeric(df['votingAuthSole'], errors='coerce').fillna(0).astype(int)
            df['votingAuthShared'] = pd.to_numeric(df['votingAuthShared'], errors='coerce').fillna(0).astype(int)
            df['votingAuthNone'] = pd.to_numeric(df['votingAuthNone'], errors='coerce').fillna(0).astype(int)
            # Ensure CUSIP is string, stripped, and uppercase
            df['cusip'] = df['cusip'].astype(str).str.strip().str.upper()
            # Additional cleaning - remove potential non-printable chars from strings
            str_cols = ['nameOfIssuer', 'titleOfClass', 'sshPrnamtType', 'investmentDiscretion']
            for col in str_cols:
                 if col in df.columns:
                      # Fill NA before regex to avoid errors
                      df[col] = df[col].fillna('').astype(str).str.replace(r'[^\x20-\x7E]+', '', regex=True).str.strip()

            logging.info(f"Successfully parsed and cleaned {len(df)} holdings from XML.")
            return df

        # Keep existing error handling for XMLSyntaxError and general Exceptions
        except etree.XMLSyntaxError as e:
            # Provide more context if possible from the error object
            line = getattr(e, 'lineno', 'N/A')
            pos = getattr(e, 'position', 'N/A')
            msg = getattr(e, 'msg', str(e))
            logging.error(f"XML Syntax Error parsing info table (Line: {line}, Pos: {pos}): {msg}")
            st.error(f"Failed to parse the XML holdings table (Line: {line}). It might be malformed. Error: {msg}")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"Unexpected error during XML parsing or DataFrame creation: {e}", exc_info=True) # Add traceback
            st.error(f"An unexpected error occurred during XML parsing: {e}")
            return pd.DataFrame()
    # ========================================================================
    # End of REVISED parse_info_table_xml method
    # ========================================================================


# --- Data Processing Functions ---

@st.cache_data(ttl=3600) # Cache results for 1 hour
def get_holdings_data(cik, year, quarter):
    """
    Orchestrates fetching and parsing 13F holdings data for a specific CIK and quarter.
    """
    handler = SECEdgarHandler(SEC_USER_AGENT)
    accession_number = handler.find_latest_13f_hr_accession(cik, year, quarter)
    if not accession_number:
        return None # Indicate failure clearly, None is better than empty DF sometimes

    xml_filename = handler.find_info_table_xml_filename(cik, accession_number)
    if not xml_filename:
         return None # Indicate failure

    xml_url = handler._get_info_table_xml_url(cik, accession_number, xml_filename)
    xml_content = handler._fetch_info_table_xml(xml_url)
    if not xml_content:
         st.error(f"Failed to download the XML file: {xml_url}")
         return None # Indicate failure

    # Parse_info_table_xml now returns df or empty df on error, or None if major issue
    parsed_df = handler.parse_info_table_xml(xml_content)
    # Return None if parsing failed critically, otherwise return the DataFrame (even if empty)
    return parsed_df


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
    Handles None inputs for dataframes.
    """
    # Handle cases where dataframes might be None
    if _df_curr is None:
         logging.error("Current holdings data is None, cannot calculate changes.")
         return pd.DataFrame() # Return empty if current data is missing
    if _df_prev is None:
         logging.warning("Previous holdings data is None. All current holdings marked as 'New'.")
         _df_prev = pd.DataFrame() # Treat as empty for calculation

    # Ensure dataframes have necessary columns, handle potential errors if parsing failed partially
    required_cols = ['cusip', 'nameOfIssuer', 'sshPrnamt', 'value']
    if not all(col in _df_curr.columns for col in required_cols):
         logging.error("Current holdings DataFrame is missing required columns for change calculation.")
         _df_curr['change_type'] = 'Error (Missing Cols)'
         # Fill missing columns with NA or default values if possible for display
         for col in required_cols + ['change_shares', 'change_value', 'change_pct']:
              if col not in _df_curr.columns:
                   _df_curr[col] = pd.NA
         return _df_curr[['cusip', 'nameOfIssuer', 'sshPrnamt', 'value', 'change_type', 'change_shares', 'change_value', 'change_pct']]

    if not _df_prev.empty and not all(col in _df_prev.columns for col in required_cols):
          logging.warning("Previous holdings DataFrame exists but missing required columns. Treating as empty.")
          _df_prev = pd.DataFrame(columns=required_cols) # Use empty df with correct columns

    # Make copies to avoid modifying cached dataframes
    df_curr_copy = _df_curr.copy()
    df_prev_copy = _df_prev.copy()

    if df_prev_copy.empty:
        # If previous quarter data is missing/empty, mark all current holdings as New
        df_curr_copy['change_type'] = 'New'
        df_curr_copy['change_shares'] = df_curr_copy['sshPrnamt']
        df_curr_copy['change_value'] = df_curr_copy['value']
        df_curr_copy['change_pct'] = 100.0 # Or pd.NA if preferred
        return df_curr_copy[['cusip', 'nameOfIssuer', 'sshPrnamt', 'value', 'change_type', 'change_shares', 'change_value', 'change_pct']]

    # Aggregate duplicate CUSIPs before merging (important for options etc.)
    # Sum numeric columns, keep first for strings
    df_curr_agg = df_curr_copy.groupby('cusip').agg(
        # Take first non-NA name, then first overall
        nameOfIssuer=('nameOfIssuer', lambda x: x.dropna().iloc[0] if not x.dropna().empty else x.iloc[0]),
        sshPrnamt=('sshPrnamt', 'sum'),
        value=('value', 'sum')
    ).reset_index()
    df_prev_agg = df_prev_copy.groupby('cusip').agg(
        nameOfIssuer=('nameOfIssuer', lambda x: x.dropna().iloc[0] if not x.dropna().empty else x.iloc[0]),
        sshPrnamt=('sshPrnamt', 'sum'),
        value=('value', 'sum')
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
        shares_curr = int(row['sshPrnamt_curr']) if pd.notna(row['sshPrnamt_curr']) else 0
        value_curr = float(row['value_curr']) if pd.notna(row['value_curr']) else 0.0
        shares_prev = int(row['sshPrnamt_prev']) if pd.notna(row['sshPrnamt_prev']) else 0
        value_prev = float(row['value_prev']) if pd.notna(row['value_prev']) else 0.0

        change_type = ''
        change_shares = 0
        change_value = 0.0
        change_pct = pd.NA # Default to NA for percentage

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
                 # Use value change % if shares are zero? Or stick to shares? Stick to shares for now.
                 change_pct = (shares_curr - shares_prev) / shares_prev * 100.0
            # else: change_pct remains pd.NA (indeterminate percentage change from zero)

            # Assign type based on share change
            if change_shares > 0:
                change_type = 'Increased'
            elif change_shares < 0:
                change_type = 'Decreased'
            else:
                 # If shares are same, check value change (price fluctuation or reporting diff)
                 if abs(value_curr - value_prev) > 1: # Check for non-trivial value change (allow for rounding)
                     change_type = 'Value Change Only'
                     # Calculate value % change if desired
                     # if value_prev != 0: change_pct = (value_curr - value_prev) / value_prev * 100.0
                 else:
                      change_type = 'Unchanged'
                      change_pct = 0.0 # Explicitly 0% change
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
    df_changes['change_pct'] = pd.to_numeric(df_changes['change_pct'], errors='coerce').fillna(pd.NA) # Allow NA

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
if month >= 11 or month < 2: # Data for Q3 current year filed by Nov 14, or if Jan check prev year Q4
    if month >= 11:
        default_q = 3
        default_year = current_year
    else: # Month is Jan
         default_q = 4
         default_year = current_year -1


# Set default year index, handling case where default_year might not be in list
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

# Use Streamlit's session state for caching fetched data? Caching is done per function call with @st.cache_data
# No explicit session state cache needed here unless we want to aggregate across runs differently.

def load_manager_data(cik_list, year, quarter):
    """Loads current and previous quarter data for multiple managers using Streamlit functions."""
    prev_year, prev_quarter = get_previous_quarter(year, quarter)
    loaded_data = {} # {cik: {'current': df, 'previous': df}}
    total_ciks = len(cik_list)
    # Use st.status for better progress indication
    with st.status(f"Loading data for {total_ciks} manager(s)...", expanded=False) as status:
        for i, cik in enumerate(cik_list):
            st.write(f"({i+1}/{total_ciks}) Loading data for CIK: {cik}...")

            # Fetch current quarter data using the cached function
            df_curr = get_holdings_data(cik, year, quarter)
            if df_curr is not None:
                 st.write(f"-> Q{quarter} {year} holdings loaded: {len(df_curr)} records.")
            else:
                 st.write(f"-> Q{quarter} {year} holdings: Failed to load or parse.")


            # Fetch previous quarter data using the cached function
            df_prev = get_holdings_data(cik, prev_year, prev_quarter)
            if df_prev is not None:
                 st.write(f"-> Q{prev_quarter} {prev_year} holdings loaded: {len(df_prev)} records.")
            else:
                 st.write(f"-> Q{prev_quarter} {prev_year} holdings: Failed to load or parse.")

            loaded_data[cik] = {'current': df_curr, 'previous': df_prev}
        status.update(label="Data loading complete!", state="complete", expanded=False)

    return loaded_data

# --- Display Logic ---
if not ciks:
    st.warning("Please enter at least one valid CIK in the sidebar.")
else:
    # Load data for all selected CIKs first
    all_manager_data = load_manager_data(ciks, selected_year, selected_quarter)

    if selected_view == "Manager View":
        st.header(f"Manager View - {selected_year} Q{selected_quarter}")
        if ciks:
            selected_cik = st.selectbox("Select Manager CIK to Display", ciks)
        else:
            selected_cik = None # Should not happen if check above works

        if selected_cik and selected_cik in all_manager_data:
            data = all_manager_data[selected_cik]
            df_curr = data['current']
            df_prev = data['previous']

            if df_curr is None: # Check if loading failed for current
                st.error(f"Could not load or parse current quarter ({selected_year} Q{selected_quarter}) data for CIK {selected_cik}. See warnings/errors during loading.")
            elif df_curr.empty:
                 # Current data loaded successfully but was empty
                 st.info(f"No holdings data was found in the filing for CIK {selected_cik} for the selected quarter ({selected_year} Q{selected_quarter}). The manager might have held no reportable assets or filed an NT (Notice) report.")
            else:
                 # Current data exists
                 prev_yr, prev_q = get_previous_quarter(selected_year, selected_quarter)
                 st.subheader(f"Holdings Changes for CIK: {selected_cik} (Q{selected_quarter} {selected_year} vs Q{prev_q} {prev_yr})")

                 if df_prev is None:
                      st.warning(f"Could not load previous quarter data for CIK {selected_cik}. Changes cannot be fully calculated. Displaying current holdings as 'New (Prev data N/A)'.")
                      df_changes = calculate_holding_changes(df_curr, None) # Pass None explicitly
                 else:
                      df_changes = calculate_holding_changes(df_curr, df_prev)

                 if not df_changes.empty:
                     st.metric("Total Holdings Reported (Current)", len(df_curr))
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


                     df_display = df_changes[[
                         'cusip', 'nameOfIssuer', 'change_type',
                         'sshPrnamt', 'value',
                         'change_shares', 'change_value', 'change_pct'
                     ]].sort_values(by=['change_type', 'value'], ascending=[True, False]).reset_index(drop=True)

                     st.dataframe(df_display.style.format({
                         'value': "${:,.0f}",
                         'change_shares': "{:+,}", # Add sign for changes
                         'change_value': "${:+,}", # Add sign for changes
                         'change_pct': "{:.1f}%"
                     }).format(precision=1, na_rep='N/A', subset=['change_pct']),
                     use_container_width=True)

                 else:
                      # Changes df is empty, but df_curr is not. Means prev data was identical or failed partially.
                      st.info(f"No changes detected or previous data unavailable/identical for CIK {selected_cik}. Displaying current holdings.")
                      st.dataframe(df_curr[['cusip', 'nameOfIssuer', 'sshPrnamt', 'value']].style.format({'value': "${:,.0f}"}), use_container_width=True)


    elif selected_view == "Stock View":
        st.header(f"Stock View - {selected_year} Q{selected_quarter}")

        if not stock_cusip_input or len(stock_cusip_input) < 8: # Allow 8 or 9
            st.warning("Please enter a valid 8 or 9-character CUSIP in the sidebar for the Stock View.")
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

                 # Calculate changes specific to this manager
                 df_changes = calculate_holding_changes(df_curr, df_prev) # Handles df_prev being None

                 # Find the specific stock in the changes df
                 stock_row = df_changes[df_changes['cusip'] == stock_cusip_input]

                 if not stock_row.empty:
                     activity = stock_row.iloc[0].to_dict()
                     activity['cik'] = cik
                     stock_activity.append(activity)
                     # Try to get a consistent stock name
                     if stock_name == "Unknown Stock" and pd.notna(activity['nameOfIssuer']):
                         stock_name = activity['nameOfIssuer']
                 # calculate_holding_changes handles exited positions if df_prev was available


            if stock_activity:
                st.subheader(f"Activity for {stock_name} (CUSIP: {stock_cusip_input})")
                df_stock_view = pd.DataFrame(stock_activity)
                # Reorder and format columns
                df_stock_display = df_stock_view[[
                     'cik', 'cusip', 'nameOfIssuer', 'change_type',
                     'sshPrnamt', 'value',
                     'change_shares', 'change_value', 'change_pct'
                ]].sort_values(by=['change_type', 'value'], ascending=[True, False]).reset_index(drop=True)

                st.dataframe(df_stock_display.style.format({
                     'value': "${:,.0f}",
                     'change_shares': "{:+,}", # Add sign
                     'change_value': "${:+,}", # Add sign
                     'change_pct': "{:.1f}%"
                }).format(precision=1, na_rep='N/A', subset=['change_pct']), # Handle NAs
                use_container_width=True)
            else:
                st.info(f"No activity found for CUSIP {stock_cusip_input} among the selected managers ({', '.join(ciks)}) for the period {selected_year} Q{selected_quarter} vs Previous.")
