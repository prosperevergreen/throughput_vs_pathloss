import './App.css';

import { useState } from 'react';
import Container from 'react-bootstrap/Container';
import Navbar from 'react-bootstrap/Navbar';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';
import Card from 'react-bootstrap/Card';
import Graph from './imgs/demo.png';

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

function App() {
  // Read SS POWER BLOCK values
  const [SSPowerBlock, setSSPowerBlock] = useState(-6);
  // This state will store the parsed data
  const [newCSVData, setNewCSVData] = useState<CSVFileType[]>([]);
  const [oldCSVData, setOldCSVData] = useState<CSVFileType[]>([]);
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (evt: React.FormEvent<HTMLFormElement>) => {
    evt.preventDefault();
    const formatedNewCSVData = getPathlossPdcp(newCSVData, SSPowerBlock);
    const formatedOldCSVData = getPathlossPdcp(oldCSVData, SSPowerBlock);
    console.log(formatedNewCSVData);
    console.log(formatedOldCSVData);
    setIsLoading(true)
    const res = await apiPostThroughputVsPathloss({
      newCSVData: formatedNewCSVData,
      oldCSVData: formatedOldCSVData,
    });
    setIsLoading(false)
    console.log(res);
    
    alert('Your data will be sent');
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
          <Row className='justify-content-center'>
            <Col md='10' lg='9' xl='8' xxl='7'>
              <Card className='p-2'>
                <Card.Img variant='top' src={Graph} className='px-5' />
                <Card.Body>
                  <Card.Title>Graph</Card.Title>
                  <Card.Text>
                    Some quick example text to build on the card title and make
                    up the bulk of the card's content.
                  </Card.Text>
                  <Button variant='primary'>Download</Button>
                </Card.Body>
              </Card>
            </Col>
          </Row>
        </Form>
      </Container>
    </div>
  );
}

export default App;
