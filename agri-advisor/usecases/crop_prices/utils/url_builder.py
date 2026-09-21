class AgmarknetURLBuilder:
    def build(self, commodity_id, state_id, commodity_name, state_name, start_date, end_date):
        district_val = "0"
        market = "0"
        district_head = "--Select--"
        market_head = "--Select--"
        url = (
            f"https://agmarknet.gov.in/SearchCmmMkt.aspx?"
            f"Tx_Commodity={commodity_id}&Tx_State={state_id}&Tx_District={district_val}&Tx_Market={market}"
            f"&DateFrom={start_date}&DateTo={end_date}"
            f"&Fr_Date={start_date}&To_Date={end_date}"
            f"&Tx_Trend=0&Tx_CommodityHead={commodity_name}&Tx_StateHead={state_name.replace(' ', '+')}"
            f"&Tx_DistrictHead={district_head}&Tx_MarketHead={market_head}"
        )
        return url 