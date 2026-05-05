import {
  ToolDescriptor,
  ReActResponse,
  ReActLoopNode,
  buildReActPrompt,
  parseReActResponse,
  executeReActStep,
} from '../ReActLoopNode.node';
import { initializeState, AgentState, serializeState } from '../../AgentStateNode/AgentStateNode.node';
import type { IExecuteFunctions } from 'n8n-workflow';

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

jest.mock('openai', () => {
  const mockCreate = jest.fn();
  return {
    __esModule: true,
    default: jest.fn().mockImplementation(() => ({
      chat: {
        completions: {
          create: mockCreate,
        },
      },
    })),
    _mockCreate: mockCreate,
  };
});

function setLLMResponse(text: string): void {
  const { _mockCreate } = jest.requireMock('openai') as { _mockCreate: jest.Mock };
  _mockCreate.mockResolvedValue({
    choices: [{ message: { content: text } }],
  });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const TOOLS: ToolDescriptor[] = [
  { name: 'sql_query', description: 'Execute SQL against the ERP database' },
  { name: 'http_get', description: 'Perform an HTTP GET request' },
];

function makeState(context = 'Retrieve Q2 sales data', maxSteps = 10): AgentState {
  return initializeState(context, maxSteps);
}

// ---------------------------------------------------------------------------
// buildReActPrompt
// ---------------------------------------------------------------------------

describe('buildReActPrompt', () => {
  it('contains the context', () => {
    const prompt = buildReActPrompt(makeState('Test context'), TOOLS);
    expect(prompt).toContain('Test context');
  });

  it('lists all tool names', () => {
    const prompt = buildReActPrompt(makeState(), TOOLS);
    expect(prompt).toContain('sql_query');
    expect(prompt).toContain('http_get');
  });

  it('shows "None" when history is empty', () => {
    const prompt = buildReActPrompt(makeState(), TOOLS);
    expect(prompt).toContain('Previous Steps:\nNone');
  });

  it('includes history when steps exist', () => {
    const { appendStep } = jest.requireActual<typeof import('../../AgentStateNode/AgentStateNode.node')>(
      '../../AgentStateNode/AgentStateNode.node',
    );
    let state = makeState();
    state = appendStep(state, {
      step: 1,
      thought: 'Need data',
      action: 'sql_query',
      actionInput: { query: 'SELECT 1' },
      observation: '1 row',
      timestamp: new Date().toISOString(),
    });
    const prompt = buildReActPrompt(state, TOOLS);
    expect(prompt).toContain('Need data');
    expect(prompt).toContain('sql_query');
    expect(prompt).toContain('1 row');
  });

  it('shows current step counter', () => {
    const prompt = buildReActPrompt(makeState(), TOOLS);
    expect(prompt).toContain('Current Step: 1 of 10');
  });

  it('instructs the model on Final Answer format', () => {
    const prompt = buildReActPrompt(makeState(), TOOLS);
    expect(prompt).toContain('Final Answer:');
  });
});

// ---------------------------------------------------------------------------
// parseReActResponse
// ---------------------------------------------------------------------------

describe('parseReActResponse', () => {
  it('parses a normal action response', () => {
    const raw =
      'Thought: I need to check the database.\n' +
      'Action: sql_query\n' +
      'Action Input: {"query": "SELECT * FROM orders"}';
    const result = parseReActResponse(raw);
    expect(result.isFinalAnswer).toBe(false);
    expect(result.thought).toBe('I need to check the database.');
    expect(result.action).toBe('sql_query');
    expect(result.actionInput).toEqual({ query: 'SELECT * FROM orders' });
  });

  it('parses a final answer response', () => {
    const raw =
      'Thought: The task is complete.\n' +
      'Final Answer: Q2 revenue was €2.4M.';
    const result = parseReActResponse(raw);
    expect(result.isFinalAnswer).toBe(true);
    expect(result.finalAnswer).toBe('Q2 revenue was €2.4M.');
    expect(result.action).toBeUndefined();
  });

  it('throws when Thought field is missing', () => {
    expect(() => parseReActResponse('Action: sql_query')).toThrow();
  });

  it('handles non-JSON action input gracefully', () => {
    const raw =
      'Thought: Try this.\n' +
      'Action: http_get\n' +
      'Action Input: not-json-at-all';
    const result = parseReActResponse(raw);
    expect(result.actionInput).toEqual({ raw: 'not-json-at-all' });
  });

  it('trims whitespace around field values', () => {
    const raw = 'Thought:   lots of spaces   \nFinal Answer:   answer here  ';
    const result = parseReActResponse(raw);
    expect(result.thought).toBe('lots of spaces');
    expect(result.finalAnswer).toBe('answer here');
  });

  it('prefers Final Answer over Action when both present', () => {
    const raw =
      'Thought: Done.\n' +
      'Action: some_tool\n' +
      'Final Answer: The answer.';
    const result = parseReActResponse(raw);
    expect(result.isFinalAnswer).toBe(true);
    expect(result.finalAnswer).toBe('The answer.');
  });

  it('returns empty actionInput when Action Input line is missing', () => {
    const raw = 'Thought: Check.\nAction: sql_query';
    const result = parseReActResponse(raw);
    expect(result.actionInput).toEqual({});
  });
});

// ---------------------------------------------------------------------------
// executeReActStep
// ---------------------------------------------------------------------------

describe('executeReActStep', () => {
  beforeEach(() => jest.clearAllMocks());

  it('returns a non-final response with updated state', async () => {
    setLLMResponse(
      'Thought: I should query the DB.\nAction: sql_query\nAction Input: {"query": "SELECT 1"}',
    );
    const state = makeState();
    const { response, updatedState } = await executeReActStep(
      state,
      TOOLS,
      '',
      'gpt-4o',
      'sk-test',
      0,
    );
    expect(response.isFinalAnswer).toBe(false);
    expect(response.action).toBe('sql_query');
    expect(updatedState.stepCount).toBe(1);
    expect(updatedState.history).toHaveLength(1);
  });

  it('returns a final answer response with updated state', async () => {
    setLLMResponse(
      'Thought: I have all the data.\nFinal Answer: Revenue is €2.4M.',
    );
    const state = makeState();
    const { response, updatedState } = await executeReActStep(
      state,
      TOOLS,
      '',
      'gpt-4o',
      'sk-test',
      0,
    );
    expect(response.isFinalAnswer).toBe(true);
    expect(response.finalAnswer).toBe('Revenue is €2.4M.');
    expect(updatedState.history[0].action).toBe('FinalAnswer');
  });

  it('records the observation in the history step', async () => {
    setLLMResponse('Thought: Continuing.\nAction: http_get\nAction Input: {}');
    const state = makeState();
    const { updatedState } = await executeReActStep(
      state,
      TOOLS,
      'Tool returned: 42 rows',
      'gpt-4o',
      'sk-test',
      0,
    );
    expect(updatedState.history[0].observation).toBe('Tool returned: 42 rows');
  });

  it('does not mutate the original state', async () => {
    setLLMResponse('Thought: Check.\nAction: sql_query\nAction Input: {}');
    const state = makeState();
    await executeReActStep(state, TOOLS, '', 'gpt-4o', 'sk-test', 0);
    expect(state.stepCount).toBe(0);
    expect(state.history).toHaveLength(0);
  });

  it('propagates LLM errors', async () => {
    const { _mockCreate } = jest.requireMock('openai') as { _mockCreate: jest.Mock };
    _mockCreate.mockRejectedValueOnce(new Error('Rate limit exceeded'));
    await expect(
      executeReActStep(makeState(), TOOLS, '', 'gpt-4o', 'sk-test', 0),
    ).rejects.toThrow('Rate limit exceeded');
  });

  it('includes the thought in the history step', async () => {
    setLLMResponse('Thought: My reasoning here.\nAction: sql_query\nAction Input: {}');
    const state = makeState();
    const { updatedState } = await executeReActStep(
      state, TOOLS, '', 'gpt-4o', 'sk-test', 0,
    );
    expect(updatedState.history[0].thought).toBe('My reasoning here.');
  });
});

// ---------------------------------------------------------------------------
// ReActLoopNode.execute() — integration tests
// ---------------------------------------------------------------------------

function makeRALNExecuteFunctions(params: Record<string, unknown>, continueOnFail = false): IExecuteFunctions {
  const defaultState = initializeState('Task context', 10);
  const serializedState = JSON.stringify(serializeState(defaultState));

  return {
    getInputData: () => [{ json: {} }],
    getNodeParameter: (name: string) => {
      if (name in params) return params[name];
      if (name === 'agentState') return serializedState;
      if (name === 'observation') return '';
      if (name === 'tools') return JSON.stringify(TOOLS);
      if (name === 'maxIterations') return 10;
      if (name === 'stopCondition') return '';
      if (name === 'humanInTheLoop') return false;
      if (name === 'temperature') return 0;
      return '';
    },
    continueOnFail: () => continueOnFail,
    getNode: () => ({ name: 'ReActLoopNode', type: 'reActLoopNode' } as never),
  } as unknown as IExecuteFunctions;
}

describe('ReActLoopNode.execute()', () => {
  const node = new ReActLoopNode();

  beforeEach(() => jest.clearAllMocks());

  it('routes to output[0] (action) for non-final response', async () => {
    setLLMResponse('Thought: Check DB.\nAction: sql_query\nAction Input: {"q":"SELECT 1"}');
    const ctx = makeRALNExecuteFunctions({
      openAiApiKey: 'sk-test',
      llmModel: 'gpt-4o',
    });
    const [actionItems, finalItems] = await node.execute.call(ctx);
    expect(actionItems.length).toBe(1);
    expect(finalItems.length).toBe(0);
    expect(actionItems[0].json.action).toBe('sql_query');
  });

  it('routes to output[1] (final) for Final Answer response', async () => {
    setLLMResponse('Thought: Done.\nFinal Answer: Revenue is €2.4M.');
    const ctx = makeRALNExecuteFunctions({
      openAiApiKey: 'sk-test',
      llmModel: 'gpt-4o',
    });
    const [actionItems, finalItems] = await node.execute.call(ctx);
    expect(finalItems.length).toBe(1);
    expect(actionItems.length).toBe(0);
    expect(finalItems[0].json.finalAnswer).toBe('Revenue is €2.4M.');
  });

  it('routes to output[1] when stopCondition regex matches', async () => {
    setLLMResponse('Thought: OK.\nAction: sql_query\nAction Input: {}');
    const ctx = makeRALNExecuteFunctions({
      openAiApiKey: 'sk-test',
      llmModel: 'gpt-4o',
      stopCondition: 'sql_query',
    });
    const [actionItems, finalItems] = await node.execute.call(ctx);
    expect(finalItems.length).toBe(1);
    expect(actionItems.length).toBe(0);
  });

  it('throws when maxIterations reached', async () => {
    const { initializeState: init, serializeState: ser } = jest.requireActual<
      typeof import('../../AgentStateNode/AgentStateNode.node')
    >('../../AgentStateNode/AgentStateNode.node');
    // Build a state already at the limit
    let fullState = init('ctx', 2);
    for (let s = 0; s < 2; s++) {
      const { appendStep } = jest.requireActual<
        typeof import('../../AgentStateNode/AgentStateNode.node')
      >('../../AgentStateNode/AgentStateNode.node');
      fullState = appendStep(fullState, {
        step: s + 1, thought: 't', action: 'a',
        actionInput: {}, observation: 'o',
        timestamp: new Date().toISOString(),
      });
    }
    const ctx = makeRALNExecuteFunctions({
      openAiApiKey: 'sk-test',
      llmModel: 'gpt-4o',
      agentState: JSON.stringify(ser(fullState)),
      maxIterations: 2,
    });
    await expect(node.execute.call(ctx)).rejects.toThrow();
  });

  it('error with continueOnFail: returns error in output[0]', async () => {
    const { _mockCreate } = jest.requireMock('openai') as { _mockCreate: jest.Mock };
    _mockCreate.mockRejectedValueOnce(new Error('API Error'));
    const ctx = makeRALNExecuteFunctions({
      openAiApiKey: 'sk-test',
      llmModel: 'gpt-4o',
    }, true);
    const [actionItems] = await node.execute.call(ctx);
    expect(actionItems[0].json.error).toBeDefined();
  });
});
