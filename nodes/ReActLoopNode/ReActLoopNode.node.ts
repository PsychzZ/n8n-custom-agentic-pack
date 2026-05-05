import {
  IExecuteFunctions,
  INodeExecutionData,
  INodeType,
  INodeTypeDescription,
  NodeOperationError,
} from 'n8n-workflow';
import OpenAI from 'openai';
import { AgentState, AgentStep, appendStep, isMaxStepsReached } from '../AgentStateNode/AgentStateNode.node';

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/**
 * @description Describes one available tool that the ReAct agent may invoke.
 * Passed into the prompt template so the LLM can select from the tool set.
 *
 * @scientific_basis Schick et al. (2023) - "Toolformer: Language Models Can
 * Teach Themselves to Use Tools" - Tools are described by name and description,
 * enabling the LLM to understand their purpose and usage without explicit
 * fine-tuning. This format aligns with the tool descriptor schema of Toolformer.
 *
 * @thesis_note DE: ToolDescriptor bildet die Toolformer-Beschreibungsschnittstelle
 * ab. Die strukturierte Beschreibung von Name und Funktion ermöglicht dem LLM,
 * das korrekte Tool ohne explizites Fine-Tuning auszuwählen.
 */
export interface ToolDescriptor {
  /** Tool identifier used in the Action field of the ReAct prompt. */
  name: string;
  /** Human-readable description of what the tool does and its inputs. */
  description: string;
}

/**
 * @description Structured representation of a parsed LLM response in the
 * ReAct Thought-Action-Observation format.
 *
 * @scientific_basis Yao et al. (2022) - "ReAct: Synergizing Reasoning and
 * Acting in Language Models" - Section 2 defines the ReAct trajectory as a
 * sequence of (thought_t, action_t, observation_t) triples. This interface
 * maps directly to one such parsed output.
 *
 * @thesis_note DE: ReActResponse kapselt das geparste LLM-Ausgabeformat. Die
 * Trennung von isFinalAnswer ermöglicht eine klare Terminierungslogik ohne
 * komplexe String-Matching-Heuristiken.
 */
export interface ReActResponse {
  /** The agent's internal reasoning step. */
  thought: string;
  /** Tool name to invoke; undefined when isFinalAnswer is true. */
  action?: string;
  /** JSON parameters for the action; undefined when isFinalAnswer is true. */
  actionInput?: Record<string, unknown>;
  /** The final answer text; defined only when isFinalAnswer is true. */
  finalAnswer?: string;
  /** True if the LLM signalled task completion via "Final Answer:". */
  isFinalAnswer: boolean;
}

// ---------------------------------------------------------------------------
// Prompt builder (independently testable)
// ---------------------------------------------------------------------------

/**
 * @description Constructs the structured ReAct prompt from the current agent
 * state and available tools. The prompt format strictly follows the trajectory
 * format defined by Yao et al. (2022).
 *
 * @scientific_basis Yao et al. (2022) - "ReAct: Synergizing Reasoning and
 * Acting in Language Models" - The prompt template in Section 2 interleaves
 * reasoning traces (thoughts) with task-specific actions, enabling the model
 * to both plan and act within a single generation step.
 *
 * @param state - The current AgentState (provides context and history).
 * @param tools - Array of available tool descriptors shown to the LLM.
 * @returns The fully assembled prompt string.
 *
 * @thesis_note DE: buildReActPrompt ist der zentrale Prompt-Baustein des
 * Artefakts. Das strenge Format (Thought / Action / Action Input / Final Answer)
 * ist essentiell für das deterministischeParsen der LLM-Ausgaben und unterscheidet
 * das Artefakt von informellen Freitext-Agenten.
 *
 * @example
 * const prompt = buildReActPrompt(state, [
 *   { name: 'sql_query', description: 'Execute SQL against the ERP database' },
 * ]);
 */
export function buildReActPrompt(state: AgentState, tools: ToolDescriptor[]): string {
  const toolList = tools
    .map((t) => `- ${t.name}: ${t.description}`)
    .join('\n');

  const historyText =
    state.history.length === 0
      ? 'None'
      : state.history
          .map(
            (s) =>
              `Step ${s.step}:\n` +
              `  Thought: ${s.thought}\n` +
              `  Action: ${s.action}\n` +
              `  Action Input: ${JSON.stringify(s.actionInput)}\n` +
              `  Observation: ${s.observation}`,
          )
          .join('\n\n');

  return `You are an agent solving a task step by step.

Context: ${state.context}

Available Tools:
${toolList}

Previous Steps:
${historyText}

Current Step: ${state.stepCount + 1} of ${state.maxSteps}

Your response MUST follow this exact format:
Thought: [your reasoning about what to do next]
Action: [tool_name]
Action Input: [JSON object with parameters]

If the task is complete, respond with:
Thought: [final reasoning]
Final Answer: [your complete answer]

Respond now:`;
}

// ---------------------------------------------------------------------------
// Response parser (independently testable)
// ---------------------------------------------------------------------------

/**
 * @description Parses a raw LLM text response into a structured ReActResponse.
 * Uses simple line-prefix matching rather than a full grammar parser, which is
 * sufficient for the constrained ReAct output format.
 *
 * @scientific_basis Yao et al. (2022) - "ReAct: Synergizing Reasoning and
 * Acting in Language Models" - The output format is strictly defined in the
 * ReAct prompt, enabling reliable regex-based parsing of the model response.
 *
 * @param raw - The raw LLM output string.
 * @returns A structured ReActResponse object.
 * @throws {Error} If the response cannot be parsed (missing Thought field).
 *
 * @thesis_note DE: parseReActResponse implementiert den Parser-Baustein des
 * ReAct-Zyklus. Die bewusst einfache Implementierung (Zeilenpräfix-Matching)
 * minimiert die Abhängigkeit von externen Parsing-Bibliotheken und ist damit
 * besser testbar und wartbar.
 *
 * @example
 * const parsed = parseReActResponse(
 *   'Thought: I need the sales data.\nAction: sql_query\nAction Input: {"query": "SELECT * FROM sales"}'
 * );
 * // parsed.thought => 'I need the sales data.'
 * // parsed.action  => 'sql_query'
 */
export function parseReActResponse(raw: string): ReActResponse {
  const lines = raw.split('\n').map((l) => l.trim());

  let thought = '';
  let action: string | undefined;
  let actionInputRaw = '';
  let finalAnswer: string | undefined;

  for (const line of lines) {
    if (line.startsWith('Thought:')) {
      thought = line.slice('Thought:'.length).trim();
    } else if (line.startsWith('Final Answer:')) {
      finalAnswer = line.slice('Final Answer:'.length).trim();
    } else if (line.startsWith('Action:')) {
      action = line.slice('Action:'.length).trim();
    } else if (line.startsWith('Action Input:')) {
      actionInputRaw = line.slice('Action Input:'.length).trim();
    }
  }

  if (!thought) {
    throw new Error(
      `Failed to parse ReAct response — missing "Thought:" field.\nRaw response:\n${raw}`,
    );
  }

  if (finalAnswer !== undefined) {
    return { thought, finalAnswer, isFinalAnswer: true };
  }

  let actionInput: Record<string, unknown> = {};
  if (actionInputRaw) {
    try {
      actionInput = JSON.parse(actionInputRaw) as Record<string, unknown>;
    } catch {
      // Non-JSON action input — wrap in a plain object
      actionInput = { raw: actionInputRaw };
    }
  }

  return { thought, action, actionInput, isFinalAnswer: false };
}

// ---------------------------------------------------------------------------
// LLM call helper
// ---------------------------------------------------------------------------

/**
 * @description Sends the ReAct prompt to the specified LLM and returns the
 * raw text response. Abstracted into a separate function to facilitate mocking
 * in tests.
 *
 * @scientific_basis Yao et al. (2022) - "ReAct: Synergizing Reasoning and
 * Acting in Language Models" - The LLM call is the central generation step in
 * the ReAct loop (f_LM in the paper's notation).
 *
 * @param prompt      - The fully assembled ReAct prompt.
 * @param model       - LLM model identifier (e.g. 'gpt-4o').
 * @param apiKey      - OpenAI API key.
 * @param temperature - Sampling temperature (0 = deterministic, recommended).
 * @returns The LLM's raw text response.
 *
 * @thesis_note DE: callLLM kapselt den API-Aufruf und ermöglicht es, das LLM
 * im Rahmen der Benchmarking-Studie auszutauschen (z.B. Claude vs. GPT-4o),
 * ohne die ReAct-Kernlogik zu ändern.
 */
export async function callLLM(
  prompt: string,
  model: string,
  apiKey: string,
  temperature: number,
): Promise<string> {
  const client = new OpenAI({ apiKey });
  const response = await client.chat.completions.create({
    model,
    temperature,
    messages: [{ role: 'user', content: prompt }],
  });
  return response.choices[0]?.message?.content ?? '';
}

// ---------------------------------------------------------------------------
// Core loop step (independently testable)
// ---------------------------------------------------------------------------

/**
 * @description Executes one ReAct iteration: builds the prompt, calls the LLM,
 * parses the response, and returns both the parsed response and the updated
 * AgentState with the new step appended.
 *
 * @scientific_basis Yao et al. (2022) - "ReAct: Synergizing Reasoning and
 * Acting in Language Models" - Section 3 describes the step-level execution
 * model where each iteration produces exactly one (thought, action, observation)
 * triple. The observation is supplied externally (by the n8n workflow) after
 * the tool executes.
 *
 * @param state       - Current AgentState before this step.
 * @param tools       - Available tool descriptors for the prompt.
 * @param observation - The observation from the previous step's tool execution
 *                      (empty string for the first step).
 * @param model       - LLM model identifier.
 * @param apiKey      - OpenAI API key.
 * @param temperature - Sampling temperature.
 * @returns Object containing the parsed response and updated state.
 *
 * @thesis_note DE: executeReActStep ist die zentrale Kernfunktion des Artefakts.
 * Sie kapselt genau einen Iteration der ReAct-Schleife und ist bewusst frei von
 * n8n-Abhängigkeiten, um vollständige Testbarkeit zu gewährleisten.
 *
 * @example
 * const { response, updatedState } = await executeReActStep(
 *   state, tools, 'Database returned 42 rows.', 'gpt-4o', apiKey, 0
 * );
 */
export async function executeReActStep(
  state: AgentState,
  tools: ToolDescriptor[],
  observation: string,
  model: string,
  apiKey: string,
  temperature: number,
): Promise<{ response: ReActResponse; updatedState: AgentState }> {
  const prompt = buildReActPrompt(state, tools);
  const rawResponse = await callLLM(prompt, model, apiKey, temperature);
  const response = parseReActResponse(rawResponse);

  if (response.isFinalAnswer) {
    // For final answers we still record the step with "FinalAnswer" as action
    const step: AgentStep = {
      step: state.stepCount + 1,
      thought: response.thought,
      action: 'FinalAnswer',
      actionInput: { answer: response.finalAnswer },
      observation: response.finalAnswer ?? '',
      timestamp: new Date().toISOString(),
    };
    return { response, updatedState: appendStep(state, step) };
  }

  // For non-final steps, record thought and action; observation will be appended
  // by the calling workflow after the tool executes.
  const step: AgentStep = {
    step: state.stepCount + 1,
    thought: response.thought,
    action: response.action ?? '',
    actionInput: response.actionInput ?? {},
    observation,
    timestamp: new Date().toISOString(),
  };

  return { response, updatedState: appendStep(state, step) };
}

// ---------------------------------------------------------------------------
// n8n Node class
// ---------------------------------------------------------------------------

/**
 * @description n8n custom node that orchestrates the ReAct (Reason + Act) loop.
 * Drives Thought → Action → Observation cycles by invoking an LLM with a
 * structured prompt and parsing the output to determine the next n8n action.
 *
 * @scientific_basis Yao et al. (2022) - "ReAct: Synergizing Reasoning and
 * Acting in Language Models" - The node implements the full ReAct execution
 * framework: it interleaves chain-of-thought reasoning (Thought) with grounded
 * tool invocations (Action) and integrates external feedback (Observation).
 *
 * @scientific_basis Schick et al. (2023) - "Toolformer: Language Models Can
 * Teach Themselves to Use Tools" - The tool descriptor format used in the prompt
 * follows the Toolformer paradigm of in-context tool description.
 *
 * @scientific_basis Wang et al. (2023) - "A Survey on Large Language Model based
 * Autonomous Agents" - Section 3.3 identifies action execution as the bridge
 * between planning and environment interaction. This node implements that bridge.
 *
 * @thesis_note DE: Der ReActLoopNode ist die Orchestrierungskomponente des
 * Artefakts. Er koordiniert AgentStateNode und VectorStoreNode und macht den
 * gesamten Entscheidungsprozess des Agenten im n8n-Workflow sichtbar —
 * im Gegensatz zu proprietären Agenten-Frameworks, die diese Logik verbergen.
 */
export class ReActLoopNode implements INodeType {
  description: INodeTypeDescription = {
    displayName: 'ReAct Loop',
    name: 'reActLoopNode',
    icon: 'fa:sync-alt',
    group: ['transform'],
    version: 1,
    description:
      'Orchestrates Thought → Action → Observation cycles using the ReAct pattern ' +
      '(Yao et al., 2022). Returns the next action or final answer for the agent.',
    defaults: {
      name: 'ReAct Loop',
      color: '#00A86B',
    },
    inputs: ['main'],
    outputs: ['main', 'main'],
    outputNames: ['Action Required', 'Final Answer'],
    properties: [
      // ------------------------------------------------------------------
      // LLM configuration
      // ------------------------------------------------------------------
      {
        displayName: 'OpenAI API Key',
        name: 'openAiApiKey',
        type: 'string',
        typeOptions: { password: true },
        default: '',
        required: true,
        description: 'OpenAI API key for the LLM calls',
      },
      {
        displayName: 'LLM Model',
        name: 'llmModel',
        type: 'options',
        options: [
          { name: 'GPT-5.4 Mini', value: 'gpt-5.4-mini' },
          { name: 'GPT-4o', value: 'gpt-4o' },
          { name: 'GPT-4o Mini', value: 'gpt-4o-mini' },
          { name: 'GPT-4 Turbo', value: 'gpt-4-turbo' },
          { name: 'GPT-3.5 Turbo', value: 'gpt-3.5-turbo' },
          { name: 'Claude Sonnet 4 (via proxy)', value: 'claude-sonnet-4-20250514' },
        ],
        default: 'gpt-4o-mini',
        description: 'The LLM to use for Thought generation (Yao et al., 2022)',
      },
      {
        displayName: 'Temperature',
        name: 'temperature',
        type: 'number',
        default: 0.0,
        typeOptions: { minValue: 0, maxValue: 1, numberPrecision: 2 },
        description: 'Sampling temperature. 0 = deterministic, recommended for tool use',
      },

      // ------------------------------------------------------------------
      // Agent state input
      // ------------------------------------------------------------------
      {
        displayName: 'Agent State (JSON)',
        name: 'agentState',
        type: 'json',
        default: '{}',
        required: true,
        description: 'Current AgentState JSON from AgentStateNode',
      },
      {
        displayName: 'Observation from Last Tool',
        name: 'observation',
        type: 'string',
        typeOptions: { rows: 3 },
        default: '',
        description:
          'The result returned by the last tool execution (empty for the first step)',
      },

      // ------------------------------------------------------------------
      // Tool registry
      // ------------------------------------------------------------------
      {
        displayName: 'Available Tools (JSON Array)',
        name: 'tools',
        type: 'json',
        default: '[]',
        description:
          'JSON array of tool descriptors: [{"name": "tool_name", "description": "..."}]. ' +
          'These are shown to the LLM so it can select the correct tool.',
      },

      // ------------------------------------------------------------------
      // Loop control
      // ------------------------------------------------------------------
      {
        displayName: 'Max Iterations',
        name: 'maxIterations',
        type: 'number',
        default: 10,
        typeOptions: { minValue: 1, maxValue: 50 },
        description:
          'Hard stop on ReAct iterations (prevents runaway loops — Wang et al., 2023). ' +
          'Should match the maxSteps in AgentStateNode.',
      },
      {
        displayName: 'Stop Condition (Regex)',
        name: 'stopCondition',
        type: 'string',
        default: '',
        description:
          'Optional regex pattern that, if matched in the LLM output, signals task completion ' +
          '(in addition to "Final Answer:").',
      },
      {
        displayName: 'Human in the Loop',
        name: 'humanInTheLoop',
        type: 'boolean',
        default: false,
        description:
          'If enabled, the node pauses after each Thought-Action pair, requiring manual ' +
          'approval before executing the action (Amershi et al., 2019).',
      },
    ],
  };

  /**
   * @description Main execution handler. Runs one ReAct iteration: reads the
   * current state, calls the LLM, parses the response, and routes output to
   * either the "Action Required" outlet or the "Final Answer" outlet.
   *
   * @scientific_basis Yao et al. (2022) - "ReAct" - The dual-output routing
   * mirrors the ReAct decision tree: either the agent needs another tool call
   * (action output) or it has reached a final answer (answer output).
   *
   * @returns Two output arrays: [actionRequired[], finalAnswer[]]
   *
   * @thesis_note DE: Die execute-Methode implementiert genau einen ReAct-
   * Iterationsschritt. Durch das Zwei-Ausgabe-Routing kann der n8n-Workflow
   * nach jedem Schritt entscheiden, ob er einen Tool-Node aufruft oder die
   * Antwort an den Nutzer weiterleitet — vollständig sichtbar im Canvas.
   */
  async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
    const items = this.getInputData();
    const actionItems: INodeExecutionData[] = [];
    const finalItems: INodeExecutionData[] = [];

    for (let i = 0; i < items.length; i++) {
      try {
        const openAiApiKey = this.getNodeParameter('openAiApiKey', i) as string;
        const llmModel = this.getNodeParameter('llmModel', i) as string;
        const temperature = this.getNodeParameter('temperature', i) as number;
        const agentStateRaw = this.getNodeParameter('agentState', i) as string;
        const observation = this.getNodeParameter('observation', i) as string;
        const toolsRaw = this.getNodeParameter('tools', i) as string;
        const maxIterations = this.getNodeParameter('maxIterations', i) as number;
        const stopCondition = this.getNodeParameter('stopCondition', i) as string;
        const humanInTheLoop = this.getNodeParameter('humanInTheLoop', i) as boolean;

        // Parse inputs
        const stateObj = typeof agentStateRaw === 'string'
          ? (JSON.parse(agentStateRaw) as Record<string, unknown>)
          : (agentStateRaw as Record<string, unknown>);

        // Import deserializeState lazily to avoid circular dep issues
        const { deserializeState, serializeState } = await import(
          '../AgentStateNode/AgentStateNode.node'
        );

        const state = deserializeState(stateObj);

        if (isMaxStepsReached(state) || state.stepCount >= maxIterations) {
          throw new NodeOperationError(
            this.getNode(),
            `ReAct loop exceeded maximum iterations (${maxIterations}). Increase maxIterations or review agent logic.`,
            { itemIndex: i },
          );
        }

        const tools: ToolDescriptor[] = toolsRaw
          ? (JSON.parse(toolsRaw) as ToolDescriptor[])
          : [];

        const { response, updatedState } = await executeReActStep(
          state,
          tools,
          observation,
          llmModel,
          openAiApiKey,
          temperature,
        );

        // Check stop condition regex
        const rawResponseText = response.finalAnswer ?? response.action ?? '';
        const isStop =
          response.isFinalAnswer ||
          (stopCondition !== '' && new RegExp(stopCondition).test(rawResponseText));

        const serialized = serializeState(updatedState);

        if (isStop) {
          finalItems.push({
            json: {
              finalAnswer: response.finalAnswer ?? rawResponseText,
              agentState: serialized,
              stepCount: updatedState.stepCount,
              humanInTheLoop,
            },
            pairedItem: { item: i },
          });
        } else {
          actionItems.push({
            json: {
              thought: response.thought,
              action: response.action,
              actionInput: response.actionInput,
              agentState: serialized,
              stepCount: updatedState.stepCount,
              humanInTheLoop,
              awaitingApproval: humanInTheLoop,
            },
            pairedItem: { item: i },
          });
        }
      } catch (error) {
        if (this.continueOnFail()) {
          actionItems.push({
            json: { error: (error as Error).message },
            pairedItem: { item: i },
          });
          continue;
        }
        throw error;
      }
    }

    return [actionItems, finalItems];
  }
}
