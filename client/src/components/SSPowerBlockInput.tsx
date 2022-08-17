import React, { useState } from 'react';
import Form from 'react-bootstrap/Form';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';

interface SSPowerBlockProps {
  setValue: React.Dispatch<React.SetStateAction<number>>;
  value: number;
  label: string;
}

const SSPowerBlockInput = ({ setValue, value, label }: SSPowerBlockProps) => {
  const [customSsPowerBlock, setCustomSsPowerBlock] = useState(false);

  const setSSPowerBlockValue = (
    value: number,
    isCustom: boolean = false
  ): void => {
    setValue(value);
    setCustomSsPowerBlock(isCustom);
  };

  return (
    <Row>
      <Col xs='12'>
        <Form.Label>
          {label}
          <span className='text-danger'>*</span>
        </Form.Label>
      </Col>
      <Col xs='12'>
        <Row>
          <Form.Group
            as={Col}
            lg='auto'
            md='12'
            xs='auto'
            className='d-flex align-items-center'>
            <Form.Check
              inline
              value={-6}
              label='-6'
              name='sspowerblock'
              type='radio'
              defaultChecked
              onChange={({ target }) =>
                setSSPowerBlockValue(Number(target.value))
              }
            />
            <Form.Check
              inline
              value={-9}
              label='-9'
              name='sspowerblock'
              type='radio'
              onChange={({ target }) =>
                setSSPowerBlockValue(Number(target.value))
              }
            />
            <Form.Check
              inline
              value={0}
              label='custom'
              name='sspowerblock'
              type='radio'
              onChange={({ target }) =>
                setSSPowerBlockValue(Number(target.value), true)
              }
            />
          </Form.Group>
          <Form.Group as={Col}>
            <Form.Control
              type='number'
              disabled={!customSsPowerBlock}
              placeholder='e.g. -12'
              value={customSsPowerBlock ? value : ''}
              onChange={({ target }) =>
                setSSPowerBlockValue(Number(target.value), true)
              }
            />
          </Form.Group>
        </Row>
      </Col>
    </Row>
  );
};

export default SSPowerBlockInput;
