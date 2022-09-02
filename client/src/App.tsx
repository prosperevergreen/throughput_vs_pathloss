import './App.css';

import { useState } from 'react';
import Container from 'react-bootstrap/Container';
import Navbar from 'react-bootstrap/Navbar';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';
import Card from 'react-bootstrap/Card';

import CSVFileInput from './components/CSVFileInput';
import SSPowerBlockInput from './components/SSPowerBlockInput';

import { CSVFileType, CSVProcessDataType } from './types';
import { apiPostThroughputVsPathloss } from './services/api';

const getPathlossPdcp = (CSVData: CSVFileType[], sspowerblock: number) => {
  if (CSVData.length === 0) return [] as CSVProcessDataType[];

  // Find and extract properties of csv which contains the 'rsrp' and 'pdcp'
  const firstEl = CSVData[0];
  const propertyName = Object.keys(firstEl).reduce(
    (accum, item) => {
      if (item.toLowerCase().includes('rsrp')) {
        return { ...accum, rsrp: item };
      }
      if (item.toLowerCase().includes('pdcp')) {
        return { ...accum, pdcp: item };
      }
      return accum;
    },
    { rsrp: '', pdcp: '' }
  );
  return CSVData.map((item) => ({
    // compute the pathloss
    // @ts-ignore
    pathloss: parseFloat(item[propertyName.rsrp]) - sspowerblock,
    // @ts-ignore
    pdcp: parseFloat(item[propertyName.pdcp]),
  }));
};

const downloadImage = (image: string) => {
  const link = document.createElement('a');
  link.href = image;
  link.setAttribute('download', 'build-compare.png'); //or any other extension
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

function App() {
  // Read SS POWER BLOCK values
  const [SSPowerBlock, setSSPowerBlock] = useState(-6);
  // This state will store the parsed data
  const [newCSVData, setNewCSVData] = useState<CSVFileType[]>([]);
  const [oldCSVData, setOldCSVData] = useState<CSVFileType[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [graphImage, setGraphImage] = useState('');

  const handleSubmit = async (evt: React.FormEvent<HTMLFormElement>) => {
    evt.preventDefault();
    // Format CSVData: {pathloss: number, pdcp: number}[]
    const formatedNewCSVData = getPathlossPdcp(newCSVData, SSPowerBlock);
    const formatedOldCSVData = getPathlossPdcp(oldCSVData, SSPowerBlock);

    // Reachout to server to process data
    setIsLoading(true);
    const res = await apiPostThroughputVsPathloss({
      newCSVData: formatedNewCSVData,
      oldCSVData: formatedOldCSVData,
    });

    // Convert png image to src type
    const responseBlob = new Blob([res.data], { type: 'image/png' });
    const reader = new window.FileReader();
    reader.onload = () => {
      setGraphImage(reader.result as string);
    };
    reader.readAsDataURL(responseBlob);
    setIsLoading(false);
  };

  return (
    <div className='App'>
      <Navbar bg='light' sticky='top'>
        <Container>
          <Navbar.Brand href='#home'>FIVE MILLIMETER WAVE</Navbar.Brand>
        </Container>
      </Navbar>
      <Container>
        <Form onSubmit={handleSubmit}>
          <Row className='my-3'>
            <Col md='5' className='mb-3' xs={{ order: 1 }}>
              <CSVFileInput setValue={setNewCSVData} label='NEW CSV' />
            </Col>
            <Col
              md={{ span: 5, offset: 2, order: 2 }}
              xs={{ order: 3 }}
              className='mb-3'>
              <SSPowerBlockInput
                label='SSPOWERBLOCK'
                setValue={setSSPowerBlock}
                value={SSPowerBlock}
              />
            </Col>
            <Col md={{ span: 5, order: 3 }} xs={{ order: 2 }} className='mb-3'>
              <CSVFileInput setValue={setOldCSVData} label='OLD CSV' />
            </Col>
            <Col
              md={{ span: 5, offset: 2 }}
              xs={{ order: 4 }}
              className='d-flex flex-column justify-content-end mb-3'>
              <Button variant='info' type='submit'>
                Submit
              </Button>
            </Col>
          </Row>
          {graphImage && (
            <Row className='justify-content-center'>
              <Col md='10' lg='9' xl='8' xxl='7'>
                <Card className='p-2'>
                  <Card.Img variant='top' src={graphImage} className='px-5' />
                  <Card.Body>
                    <Card.Title>Graph</Card.Title>
                    <Card.Text>
                      Comparison between the selected software build of the throughput vs pathloss.
                    </Card.Text>
                    <Button
                      variant='primary'
                      onClick={(_e) => downloadImage(graphImage)}>
                      Download
                    </Button>
                  </Card.Body>
                </Card>
              </Col>
            </Row>
          )}
        </Form>
      </Container>
    </div>
  );
}

export default App;
