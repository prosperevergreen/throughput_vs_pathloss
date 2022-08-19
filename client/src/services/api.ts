import axios from 'axios';
import { CSVProcessDataType } from '../types';

interface PostThroughputVsPathloss {
  oldCSVData: CSVProcessDataType[];
  newCSVData: CSVProcessDataType[];
}

export const apiPostThroughputVsPathloss = (data: PostThroughputVsPathloss) =>
  axios
    .post(
      'http://localhost:5005/api/throughput-vs-pathloss',
      data
    )
    .then((res) => res);

// fetch("https://google.com", {
//   method: "POST",
//   mode: "cors",
//   headers: {
//       // Authorization: `Bearer: ${token}`,
//       "Content-Type": "application/json",
//   },
//   body: JSON.stringify(data),
// }).then((res)=> res.json());
