from bs4 import BeautifulSoup
from typing import List, Dict

class AgmarknetHTMLParser:
    def parse(self, html: str) -> List[Dict]:
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find('table', id='cphBody_GridPriceData')
        if not table:
            return []
        rows = table.find_all('tr')
        if not rows or len(rows) < 2:
            return []
        header_cells = rows[0].find_all(['th', 'td'])
        headers = [cell.get_text(strip=True).replace(" ", "_").replace("(", "").replace(")", "") for cell in header_cells]
        json_list = []
        for i, row in enumerate(rows[1:], 1):
            cells = row.find_all(['td', 'th'])
            if len(cells) < 2:
                continue
            if 'No Data Found' in row.get_text():
                continue
            record = {headers[j] if j < len(headers) else f"col{j}": cells[j].get_text(strip=True) for j in range(len(cells))}
            record['Sl_No'] = str(i)
            json_list.append(record)
        return json_list