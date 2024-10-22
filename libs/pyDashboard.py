import requests
import json
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from libs.apiConfig import ApiConfig


class pyDashboard(object):
    # --- constructor ---
    def __init__(self, api, apikey):
        self.__api_url_base = ApiConfig.base_url
        self.__api_key = ApiConfig.api_key

        self.__init_requests()

        # populates dict with id's and respective sensor, metric and unit names
        self.__init_names()

    #  --- Sends a request for data and returns the raw json file  ---

    def request_data(self, req, sensor_ids=None, last_n_mins=None, days=None, metric=None, log=False):

        # Forms url for intended request
        if req is not None and sensor_ids is None and last_n_mins is None and days is None and metric is None:
            api_url = f'{self.__api_url_base}{req}'
            print(api_url)

        elif req == self.__reqs.history and sensor_ids is not None and last_n_mins is not None:
            api_url = f'{self.__api_url_base}{req}?sensor={sensor_ids}&minutes={last_n_mins}'

        elif req == self.__reqs.history and sensor_ids is not None and days is not None:
            api_url = f'{self.__api_url_base}{req}?sensor={sensor_ids}&days={days}'

        else:
            print("ERR in pyDashboard -> get_data():\n  ")
            return None

        # Creates header and sends request
        headers = {
            'accept': 'application/json',
            'ApiKey': f'{self.__api_key}',
        }
        try:
            response = requests.get(api_url, headers=headers)
            SC = response.status_code
        except Exception as e:
            print(f"ERR in pyDashboard -> get_data():\n  {e}")
            return None

        # IF log THEN Outputs the response status code
        if log:
            print(
                f"pyDashboard -> \n  Request: GET {api_url}\n  Response: {SC} - {self.__status_code_name[SC]}")

        # Returns the data or None
        if SC == 200:
            return json.loads(response.text)
        else:
            return None

    # --- Initializes requests ---

    def __init_requests(self):
        nontupl = ()
        for i in range(len(self.__All_requests.__annotations__)):
            nontupl += (None,)
        print(nontupl)
        self.__reqs = self.__All_requests(*nontupl)

    def get_reqs(self):
        return self.__reqs

    # --- Requests all sensors, metrics and units from server and populates dicts ---
    def __init_names(self):
        json_data = self.request_data(self.__reqs.metrics)
        df = pd.json_normalize(json_data["metrics"])
        df = df.reset_index()
        for _, row in df.iterrows():
            self.__metrics_names[row['id']] = row['name']

        df = pd.json_normalize(json_data['metrics'],
                               record_path=['units'])

        df = df.reset_index()
        for _, row in df.iterrows():
            self.__unit_names[row['id']] = row['name']

        json_data = self.request_data(self.__reqs.sensors)
        df = pd.json_normalize(json_data['sensors'])
        df = df.reset_index()
        for _, row in df.iterrows():
            self.__sensor_names[row['id']] = row['name']

    # --- Packs the json file into a Pandas data frame ---
    def get_df(self, req, sensor_ids=None, last_n_mins=None, days=None, all_sensors=None, metric=None, log=False):

        # Handles base requests
        if req is not None and sensor_ids is None and last_n_mins is None and days is None and metric is None:
            json_data = self.request_data(req, log=log)
            # print(json_data)

        # Handles multiple sensor history requests
        elif req is self.__reqs.history and last_n_mins is not None:

            if sensor_ids is not None:

                if type(sensor_ids) is list:
                    sensor_ids = '%2C'.join(sensor_ids)

            elif all_sensors:
                sensor_ids = '%2C'.join(self.__sensor_names.keys())

            json_data = self.request_data(
                req, sensor_ids, last_n_mins, metric=metric, log=log)

            # Handles history requests with metric filter
            api_url = '{0}{1}?sensor={2}&minutes={3}'.format(
                self.__api_url_base, req, sensor_ids, last_n_mins
            )

            if metric is not None:
                api_url += f'&metric={metric}'

            json_data = self.request_data(api_url, log=log)

        # Handles multiple sensor history requests
        elif req is self.__reqs.history and days is not None:

            if sensor_ids is not None:

                if type(sensor_ids) is list:
                    sensor_ids = '%2C'.join(sensor_ids)

            elif all_sensors:
                sensor_ids = '%2C'.join(self.__sensor_names.keys())

            json_data = self.request_data(
                req, sensor_ids, days=days, log=log)
            # print(json_data)

        # Returns None if arguments don't match
        else:
            print(req)
            print("ERR in pyDashboard -> get_data():\n  ")
            return None
        # Returns None if json is empty
        if json_data == None:
            print("get_df() failed")
            return None

        # Formats the sensors DF
        elif req == self.__reqs.sensors:
            print(
                "INF pyDashboard -> get_df():\n  packing 'sensors', dropping 'links'")
            df = pd.json_normalize(json_data['sensors'],
                                   record_path=['skills'],
                                   meta=[
                'id',
                'sensorId',
                'name',
                'type',
                'bases',
            ])

        # Formats the metrics DF
        elif req == self.__reqs.metrics:
            df = pd.json_normalize(json_data['metrics'],
                                   record_path=['units'],
                                   meta=[
                'id',
                'name',
            ],
                record_prefix='unit_')

            print(
                "INF pyDashboard -> get_df():\n  packing 'metrics', dropping not-default")
            df = df.dropna()

        # Formats the bases DF
        elif req == self.__reqs.bases:
            df = pd.json_normalize(json_data['bases'])

        # Formats the history DF
        elif req == self.__reqs.history:
            try:
                df = pd.json_normalize(json_data['readings'])
            except Exception as e:
                print(
                    "ERR in pyDashboard -> get_df():\n  {0} - try increasing time frame (no measurements received)".format(e))
                if log:
                    print("sensor_ids: {0}".format(sensor_ids))
                return None

        return df
        # --- Adds sensor locations to the DF ---

    def add_locations_to_df(self, df):
        if df is not None:
            df['type'] = df['sensor'].apply(self.__get_stype)
            df['floor'] = df['sensor'].apply(self.__get_sfloor)
            df['room'] = df['sensor'].apply(self.__get_sroom)
            return df
        else:
            print("ERR  add_locations_to_df -> provided df is None!")
            return None

    # --- Returns the available requests for dashboard ---

    def get_reqs(self):
        return self.__reqs

    # --- Prints or fills a list with available sensors ---
    def available_sensors(self, log=True):
        # IF log THEN prints available sensors to command line
        if log:
            print("available sensors:")
            if len(self.__sensor_names) != 0:
                for k, v in self.__sensor_names.items():
                    print("  ", k, v)
            else:
                print("None")

        # If a list is provided, it gets filled with sensor names
        return self.__sensor_names

    # --- Converts metric ids to names ---
    def metric_name(self, id):
        return self.__metrics_names[id]

    # Inside the pyDashboard class
    def sensor_name(self, id):
        return self.__sensor_names.get(id, id)

    # --- Converts unit ids to names ---

    def unit_name(self, id):
        return self.__unit_names[id]

    # --- Extracts date from stamp ---
    def extract_date(self, date_time):
        # 2023-01-13T15:27:32Z
        return date_time.strftime("%Y-%m-%d")

    # --- Extracts time from stamp ---
    def extract_time(self, date_time):
        return date_time.strftime("%H:%M:%S")

    def convert_timezone(self, date_time):
        dt = datetime.strptime(date_time, "%Y-%m-%dT%H:%M:%SZ")
        dt = dt.replace(tzinfo=timezone.utc).astimezone(
            tz=timezone(timedelta(hours=2, minutes=0), "Europe/Riga"))
        return dt

    # --- Prints or fills a list with available sensors ---
    def available_sensors(self, log=True):
        # IF log THEN prints available sensors to command line
        if log:
            print("available sensors:")
            if len(self.__sensor_names) != 0:
                for k, v in self.__sensor_names.items():
                    print("  ", k, v)
            else:
                print("None")

        # If a list is provided, it gets filled with sensor names
        return self.__sensor_names

    # --- Makes a sensor message readable by replacing ids with names ---
    def make_sensor_df_readable(self, df, include_seperate_date_and_time=True):
        if df is not None:
            df['sensorid'] = df['sensor']
            df['metric'] = df['metric'].apply(self.metric_name)
            df['sensor'] = df['sensor'].apply(self.sensor_name)
            df['unit'] = df['unit'].apply(self.unit_name)
            if include_seperate_date_and_time:
                df['datetime'] = df['time'].apply(self.convert_timezone)
                df['date'] = df['datetime'].apply(self.extract_date)
                df['time'] = df['datetime'].apply(self.extract_time)
            else:
                df['time'] = df['time'].apply(self.convert_timezone)

        else:
            print("ERR  make_sensor_df_readable -> provided df is None!")
        return df

    #  PRIVATE VARIABLES

    __api_url_base = ''
    __api_key = ''
    __status_code_name = {200: "Request OK", 404: "Not Found",
                          500: "Internal server error", 400: "Bad Request"}
    __metrics_names = dict()
    __unit_names = dict()
    __sensor_names = dict()

    @dataclass
    class __All_requests:
        bases: str
        metrics: str
        sensors: str
        history: str

        def __post_init__(self):
            if self.bases is None:
                self.bases = 'v1/bases'
            if self.metrics is None:
                self.metrics = 'v1/metrics'
            if self.sensors is None:
                self.sensors = 'v1/sensors'
            if self.history is None:
                self.history = 'v1/measurements/history'
