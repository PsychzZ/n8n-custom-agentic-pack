import {
  IDataObject,
  IExecuteFunctions,
  INodeExecutionData,
  INodeType,
  INodeTypeDescription,
  NodeOperationError,
} from 'n8n-workflow';
import { v4 as uuidv4 } from 'uuid';

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/**
 * @description Represents a single completed Thought → Action → Observation cycle
 * within a ReAct-style agent loop. Each step is an immutable record appended
 * to the agent's memory stream.
 *
 * @scientific_basis Park et al. (2023) - "Generative Agents: Interactive Simulacra
 * of Human Behavior" - The authors introduce the concept of a memory stream as a
 * comprehensive, append-only log of an agent's experience. Each entry carries a
 * timestamp to support temporal reasoning and retrieval.
 *
 * @scientific_basis Yao et al. (2022) - "ReAct: Synergizing Reasoning and Acting
 * in Language Models" - The Thought/Action/Observation triple maps directly to the
 * ReAct trajectory format described in Section 2 of the paper.
 *
 * @thesis_note DE: AgentStep bildet die atomare Einheit des ReAct-Zyklus ab. Die
 * strukturierte Speicherung von Thought, Action und Observation ermöglicht sowohl
 * die Nachvollziehbarkeit einzelner Entscheidungen als auch die spätere maschinelle
 * Auswertung im Rahmen der Benchmarking-Studie (vgl. Evaluation, Kapitel 6).
 */
export interface AgentStep {
  /** Sequential step index within the current session (1-based). */
  step: number;
  /** The agent's internal reasoning before selecting an action. */
  thought: string;
  /** The tool name or action identifier chosen by the agent. */
  action: string;
  /** Structured parameters passed to the selected action/tool. */
  actionInput: Record<string, unknown>;
  /** The result returned by the tool or external environment. */
  observation: string;
  /** ISO 8601 creation timestamp for temporal ordering and benchmarking. */
  timestamp: string;
}

/**
 * @description Top-level state container for a single agent session. Implements
 * the working-memory abstraction described by Park et al. (2023), extended with
 * loop-termination guards from Wang et al. (2023).
 *
 * @scientific_basis Park et al. (2023) - "Generative Agents" - Working memory
 * is modelled as a bounded memory stream. The sessionId enables isolation between
 * concurrent agent instances, analogous to the agent identity construct in the paper.
 *
 * @scientific_basis Wang et al. (2023) - "A Survey on Large Language Model based
 * Autonomous Agents" - Section 3.2 discusses safety mechanisms including hard
 * iteration caps (maxSteps) to prevent runaway loops in production deployments.
 *
 * @thesis_note DE: Die AgentState-Schnittstelle ist das zentrale Datenmodell des
 * Artefakts. Die Trennung von context (aktuelle Aufgabe) und history (vergangene
 * Schritte) folgt dem Dual-Memory-Modell aus Park et al. (2023) und ermöglicht
 * eine klare Abgrenzung zwischen Kurz- und Langzeitgedächtnis des Agenten.
 */
export interface AgentState {
  /** UUID-v4 identifier uniquely scoping this run. */
  sessionId: string;
  /** Natural-language description of the current task. */
  context: string;
  /** Ordered list of all completed ReAct steps (memory stream). */
  history: AgentStep[];
  /** Number of steps executed so far (mirrors history.length for fast access). */
  stepCount: number;
  /** Hard upper bound on iterations; enforced before each new step. */
  maxSteps: number;
  /** Arbitrary domain-specific key-value data attached to this session. */
  metadata: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Pure business-logic functions (independently testable)
// ---------------------------------------------------------------------------

/**
 * @description Creates a fresh, zero-step AgentState. Generates a new UUID v4
 * session identifier so that concurrent agent runs never share state.
 *
 * @scientific_basis Park et al. (2023) - "Generative Agents" - Each simulated
 * agent is initialized with an identity and an empty memory stream before
 * receiving its first percept.
 *
 * @param context  - Natural-language description of the task to be solved.
 * @param maxSteps - Maximum number of ReAct iterations allowed (default 10).
 * @param metadata - Optional domain-specific key-value data.
 * @returns A new AgentState with an empty history.
 *
 * @thesis_note DE: Die initializeState-Funktion entspricht dem "Bootstrapping"-
 * Schritt eines Agenten. Sie legt die Session-ID als eindeutigen Schlüssel fest,
 * unter dem der Zustand in n8n-Workflow-Variablen oder einem externen KV-Speicher
 * persistiert werden kann.
 *
 * @example
 * const state = initializeState('Retrieve Q2 sales figures from ERP', 15);
 * // state.sessionId => 'f47ac10b-58cc-4372-a567-0e02b2c3d479'
 * // state.history   => []
 * // state.stepCount => 0
 */
export function initializeState(
  context: string,
  maxSteps = 10,
  metadata: Record<string, unknown> = {},
): AgentState {
  return {
    sessionId: uuidv4(),
    context,
    history: [],
    stepCount: 0,
    maxSteps,
    metadata,
  };
}

/**
 * @description Appends a completed ReAct step to the agent's history array,
 * implementing a bounded working memory. Oldest steps are NOT pruned by default;
 * truncation strategy must be configured via maxHistorySteps (future extension).
 *
 * @scientific_basis Park et al. (2023) - "Generative Agents" - The authors
 * demonstrate that maintaining a sequential memory stream with timestamps enables
 * coherent multi-step behaviour. This implementation follows their memory-stream
 * model by treating history as an append-only log.
 *
 * @param state - Current AgentState object (treated as immutable — returns a copy).
 * @param step  - The completed AgentStep to append.
 * @returns A new AgentState with the step appended and stepCount incremented.
 *
 * @thesis_note DE: Diese Funktion implementiert den "Memory Stream"-Ansatz aus
 * Park et al. (2023). Im Gegensatz zu einem einfachen String-Kontext ermöglicht
 * die strukturierte History eine spätere Analyse der Entscheidungsschritte —
 * ein zentrales Qualitätsmerkmal gegenüber autonomen Blackbox-Agenten.
 *
 * @example
 * const updated = appendStep(state, {
 *   step: 1,
 *   thought: 'I need to check the database first.',
 *   action: 'sql_query',
 *   actionInput: { query: 'SELECT * FROM orders WHERE status = \'pending\'' },
 *   observation: 'Returned 42 rows.',
 *   timestamp: new Date().toISOString(),
 * });
 */
export function appendStep(state: AgentState, step: AgentStep): AgentState {
  return {
    ...state,
    history: [...state.history, step],
    stepCount: state.stepCount + 1,
  };
}

/**
 * @description Checks whether the agent has reached its configured iteration
 * limit. Must be called before invoking the LLM in each cycle.
 *
 * @scientific_basis Wang et al. (2023) - "A Survey on Large Language Model based
 * Autonomous Agents" - Section 3.2 identifies runaway loops as a critical failure
 * mode in production agent deployments. Hard step caps are the recommended
 * mitigation.
 *
 * @param state - The current AgentState to evaluate.
 * @returns `true` if stepCount has reached or exceeded maxSteps.
 *
 * @thesis_note DE: Die Terminierungsprüfung ist ein essenzieller Sicherheitsmechanismus.
 * Sie verhindert, dass ein fehlerhafter Prompt den Agenten in eine Endlosschleife
 * treibt und schützt so vor unkontrollierten API-Kosten und Laufzeitfehlern.
 *
 * @example
 * if (isMaxStepsReached(state)) {
 *   throw new Error('Agent exceeded maximum iteration count');
 * }
 */
export function isMaxStepsReached(state: AgentState): boolean {
  return state.stepCount >= state.maxSteps;
}

/**
 * @description Resets an existing AgentState to zero steps while preserving the
 * sessionId, context, maxSteps, and metadata. Useful for retrying a failed run
 * without creating a new session.
 *
 * @scientific_basis Park et al. (2023) - "Generative Agents" - Agents can be
 * "re-spawned" with the same identity after a crash, which requires preserving
 * the session identifier while clearing the ephemeral memory stream.
 *
 * @param state - The AgentState to reset.
 * @returns A new AgentState with an empty history and stepCount 0.
 *
 * @thesis_note DE: Die Reset-Operation unterstützt den SC-03-Benchmark (Error
 * Recovery). Sie erlaubt es, denselben Agenten nach einem erzwungenen Fehler
 * neu zu starten, ohne den übergeordneten Workflow neu initialisieren zu müssen.
 *
 * @example
 * const fresh = resetState(brokenState);
 * // fresh.sessionId === brokenState.sessionId  => true (identity preserved)
 * // fresh.history                              => []
 */
export function resetState(state: AgentState): AgentState {
  return {
    ...state,
    history: [],
    stepCount: 0,
  };
}

/**
 * @description Serialises an AgentState to a plain JSON-compatible object for
 * storage in n8n workflow data or an external key-value store.
 *
 * @param state - The AgentState to serialise.
 * @returns A JSON-safe representation of the state.
 *
 * @thesis_note DE: Die Serialisierung ermöglicht die persistente Speicherung des
 * Agentenzustands zwischen n8n-Node-Ausführungen, was das zentrale Merkmal der
 * zustandsorientierten Architektur gegenüber zustandslosen Ansätzen darstellt.
 */
export function serializeState(state: AgentState): Record<string, unknown> {
  return JSON.parse(JSON.stringify(state)) as Record<string, unknown>;
}

/**
 * @description Deserialises a plain object (e.g. from n8n workflow data) back
 * into a typed AgentState, validating required fields.
 *
 * @param raw - The raw object to deserialise.
 * @returns A validated AgentState.
 * @throws {Error} If required fields (sessionId, context, history) are missing.
 *
 * @thesis_note DE: Die Deserialisierung mit Validierung stellt sicher, dass
 * korrumpierte oder unvollständige Zustandsobjekte frühzeitig erkannt werden,
 * bevor sie zu schwer debuggbaren Laufzeitfehlern führen.
 */
export function deserializeState(raw: Record<string, unknown>): AgentState {
  if (
    typeof raw.sessionId !== 'string' ||
    typeof raw.context !== 'string' ||
    !Array.isArray(raw.history)
  ) {
    throw new Error(
      'Invalid AgentState: missing required fields (sessionId, context, history)',
    );
  }
  return {
    sessionId: raw.sessionId,
    context: raw.context,
    history: raw.history as AgentStep[],
    stepCount: typeof raw.stepCount === 'number' ? raw.stepCount : (raw.history as AgentStep[]).length,
    maxSteps: typeof raw.maxSteps === 'number' ? raw.maxSteps : 10,
    metadata: typeof raw.metadata === 'object' && raw.metadata !== null
      ? (raw.metadata as Record<string, unknown>)
      : {},
  };
}

// ---------------------------------------------------------------------------
// n8n Node class
// ---------------------------------------------------------------------------

/**
 * @description n8n custom node that manages persistent AgentState across workflow
 * executions. Supports four operations (initialize / update / get / reset)
 * enabling stateful multi-step agent loops directly within n8n workflows.
 *
 * @scientific_basis Park et al. (2023) - "Generative Agents: Interactive Simulacra
 * of Human Behavior" - The node implements the memory stream concept, providing
 * structured, queryable history storage for LLM-driven agents.
 *
 * @scientific_basis Wang et al. (2023) - "A Survey on Large Language Model based
 * Autonomous Agents" - Section 4 identifies persistent state as a precondition for
 * reliable multi-step task execution. This node closes the statefulness gap in
 * standard n8n workflow orchestration.
 *
 * @thesis_note DE: Der AgentStateNode ist das Kernelement der zustandsorientierten
 * Architektur. Er ermöglicht es, den Zustand eines Agenten explizit im Workflow
 * zu verwalten und macht den Entscheidungsprozess vollständig transparent und
 * auditierbar — im Gegensatz zu proprietären Agenten-Frameworks.
 */
export class AgentStateNode implements INodeType {
  description: INodeTypeDescription = {
    displayName: 'Agent State',
    name: 'agentStateNode',
    icon: 'fa:brain',
    group: ['transform'],
    version: 1,
    description:
      'Manages persistent agent state (context, history, step count) for multi-step ReAct loops. ' +
      'Based on the memory stream model of Park et al. (2023).',
    defaults: {
      name: 'Agent State',
      color: '#7B68EE',
    },
    inputs: ['main'],
    outputs: ['main'],
    properties: [
      // ------------------------------------------------------------------
      // Operation selector
      // ------------------------------------------------------------------
      {
        displayName: 'Operation',
        name: 'operation',
        type: 'options',
        noDataExpression: true,
        options: [
          {
            name: 'Initialize',
            value: 'initialize',
            description: 'Create a new AgentState for a session',
            action: 'Initialize a new agent state',
          },
          {
            name: 'Update',
            value: 'update',
            description: 'Append a completed ReAct step to the history',
            action: 'Update agent state with a new step',
          },
          {
            name: 'Get',
            value: 'get',
            description: 'Retrieve the current state (pass-through with validation)',
            action: 'Get current agent state',
          },
          {
            name: 'Reset',
            value: 'reset',
            description: 'Clear history while preserving session identity',
            action: 'Reset agent state history',
          },
        ],
        default: 'initialize',
      },

      // ------------------------------------------------------------------
      // initialize fields
      // ------------------------------------------------------------------
      {
        displayName: 'Context',
        name: 'context',
        type: 'string',
        typeOptions: { rows: 4 },
        default: '',
        required: true,
        displayOptions: { show: { operation: ['initialize'] } },
        description: 'Natural-language description of the task this agent session should solve',
      },
      {
        displayName: 'Max Steps',
        name: 'maxSteps',
        type: 'number',
        default: 10,
        required: true,
        displayOptions: { show: { operation: ['initialize'] } },
        description: 'Hard upper bound on ReAct iterations to prevent infinite loops (Wang et al., 2023)',
        typeOptions: { minValue: 1, maxValue: 100 },
      },
      {
        displayName: 'Metadata (JSON)',
        name: 'metadata',
        type: 'json',
        default: '{}',
        displayOptions: { show: { operation: ['initialize'] } },
        description: 'Optional JSON object with domain-specific key-value data attached to this session',
      },

      // ------------------------------------------------------------------
      // update fields
      // ------------------------------------------------------------------
      {
        displayName: 'Current State (JSON)',
        name: 'currentState',
        type: 'json',
        default: '{}',
        required: true,
        displayOptions: { show: { operation: ['update', 'get', 'reset'] } },
        description: 'The AgentState JSON object from a previous Initialize or Update operation',
      },
      {
        displayName: 'Thought',
        name: 'thought',
        type: 'string',
        typeOptions: { rows: 3 },
        default: '',
        required: true,
        displayOptions: { show: { operation: ['update'] } },
        description: "The agent's internal reasoning before selecting the action (Yao et al., 2022)",
      },
      {
        displayName: 'Action',
        name: 'action',
        type: 'string',
        default: '',
        required: true,
        displayOptions: { show: { operation: ['update'] } },
        description: 'The tool name or action identifier chosen by the agent',
      },
      {
        displayName: 'Action Input (JSON)',
        name: 'actionInput',
        type: 'json',
        default: '{}',
        required: true,
        displayOptions: { show: { operation: ['update'] } },
        description: 'Structured parameters passed to the action/tool as a JSON object',
      },
      {
        displayName: 'Observation',
        name: 'observation',
        type: 'string',
        typeOptions: { rows: 3 },
        default: '',
        required: true,
        displayOptions: { show: { operation: ['update'] } },
        description: 'The result returned by the tool or external environment',
      },
    ],
  };

  /**
   * @description Entry point called by n8n's execution engine for each incoming
   * item. Dispatches to the appropriate operation handler based on the selected
   * operation parameter.
   *
   * @scientific_basis Park et al. (2023) - "Generative Agents" - The execute
   * method mirrors the agent perception-action loop: the node receives an input
   * item (percept), processes it against the current memory state, and emits
   * the updated state downstream.
   *
   * @returns Array of output item arrays (standard n8n return shape).
   *
   * @thesis_note DE: Die execute-Methode ist der Einstiegspunkt für den n8n-
   * Ausführungsmechanismus. Sie implementiert das Dispatch-Muster, um die
   * verschiedenen Zustandsoperationen hinter einer einheitlichen Node-Schnittstelle
   * zu kapseln.
   */
  async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
    const items = this.getInputData();
    const returnData: INodeExecutionData[] = [];

    for (let i = 0; i < items.length; i++) {
      try {
        const operation = this.getNodeParameter('operation', i) as string;
        let outputState: AgentState;

        switch (operation) {
          case 'initialize': {
            const context = this.getNodeParameter('context', i) as string;
            const maxSteps = this.getNodeParameter('maxSteps', i) as number;
            const metadataRaw = this.getNodeParameter('metadata', i) as string;
            const metadata = metadataRaw
              ? (JSON.parse(metadataRaw) as Record<string, unknown>)
              : {};
            outputState = initializeState(context, maxSteps, metadata);
            break;
          }

          case 'update': {
            const rawState = this.getNodeParameter('currentState', i) as string;
            const stateObj = typeof rawState === 'string'
              ? (JSON.parse(rawState) as Record<string, unknown>)
              : (rawState as Record<string, unknown>);
            const currentState = deserializeState(stateObj);

            if (isMaxStepsReached(currentState)) {
              throw new NodeOperationError(
                this.getNode(),
                `Agent has reached maximum steps (${currentState.maxSteps}). Reset the state or increase maxSteps.`,
                { itemIndex: i },
              );
            }

            const thought = this.getNodeParameter('thought', i) as string;
            const action = this.getNodeParameter('action', i) as string;
            const actionInputRaw = this.getNodeParameter('actionInput', i) as string;
            const actionInput = actionInputRaw
              ? (JSON.parse(actionInputRaw) as Record<string, unknown>)
              : {};
            const observation = this.getNodeParameter('observation', i) as string;

            const step: AgentStep = {
              step: currentState.stepCount + 1,
              thought,
              action,
              actionInput,
              observation,
              timestamp: new Date().toISOString(),
            };

            outputState = appendStep(currentState, step);
            break;
          }

          case 'get': {
            const rawState = this.getNodeParameter('currentState', i) as string;
            const stateObj = typeof rawState === 'string'
              ? (JSON.parse(rawState) as Record<string, unknown>)
              : (rawState as Record<string, unknown>);
            outputState = deserializeState(stateObj);
            break;
          }

          case 'reset': {
            const rawState = this.getNodeParameter('currentState', i) as string;
            const stateObj = typeof rawState === 'string'
              ? (JSON.parse(rawState) as Record<string, unknown>)
              : (rawState as Record<string, unknown>);
            const stateToReset = deserializeState(stateObj);
            outputState = resetState(stateToReset);
            break;
          }

          default:
            throw new NodeOperationError(
              this.getNode(),
              `Unknown operation: ${operation}`,
              { itemIndex: i },
            );
        }

        returnData.push({
          json: serializeState(outputState) as IDataObject,
          pairedItem: { item: i },
        });
      } catch (error) {
        if (this.continueOnFail()) {
          returnData.push({
            json: { error: (error as Error).message },
            pairedItem: { item: i },
          });
          continue;
        }
        throw error;
      }
    }

    return [returnData];
  }
}
